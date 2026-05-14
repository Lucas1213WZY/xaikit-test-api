"""
CoAX Forward Reasoning Strategies

Faithful re-implementation of code_for_papers/old/coax/gcm_multiple_strategies.py.

Key properties preserved from the paper:
  - ACT-R temporal decay:  activation = -decay_param * ln(time_elapsed)
  - Retrieval threshold:   only exemplars with activation >= threshold returned
  - GCM similarity:        sim = exp((-sensitivity * dist + activation) / temperature)
  - Distance:              Euclidean with diffs clipped to [-1, 1], normalised by sqrt(n_valid_dims)
  - Logical trial time:    advances by a constant per trial (not wall-clock time)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..interface import (
    ReasoningMode,
    ReasoningStrategy,
    StrategyConfig,
    StrategyMetadata,
    StrategyType,
)


# ---------------------------------------------------------------------------
# Internal timer – mirrors the external time-manager used in the paper code.
# Each strategy creates its own instance so trials advance independently.
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self) -> None:
        self._t = 0.0

    def get_time(self) -> float:
        return self._t

    def add_time(self, delta: float) -> None:
        self._t += delta


# ---------------------------------------------------------------------------
# Memory (mirrors MemoryBase from gcm_multiple_strategies.py exactly)
# ---------------------------------------------------------------------------

class _MemoryBase:
    """
    activation = -decay_param * ln(time_elapsed)
    Exemplars below retrieval_threshold are silently skipped on retrieval.
    """

    def __init__(self, decay_param: float = 0.5, retrieval_threshold: Optional[float] = None) -> None:
        self._exemplars: List[Dict] = []
        self.decay_param = decay_param
        self.retrieval_threshold = retrieval_threshold

    def store(self, label_probs: Dict, features, explanation, time_stored: float) -> None:
        self._exemplars.append({
            "label_probs":  copy.deepcopy(label_probs),
            "features":     copy.deepcopy(features),
            "explanation":  copy.deepcopy(explanation),
            "time_stored":  time_stored,
        })

    def retrieve(self, current_time: float) -> List[Tuple[Dict, float]]:
        out = []
        for ex in self._exemplars:
            elapsed = max(1e-12, current_time - ex["time_stored"])
            activation = -self.decay_param * math.log(elapsed)
            if self.retrieval_threshold is not None and activation < self.retrieval_threshold:
                continue
            out.append((ex, activation))
        return out

    def size(self) -> int:
        return len(self._exemplars)


# ---------------------------------------------------------------------------
# Utility functions (mirrors gcm_multiple_strategies.py exactly)
# ---------------------------------------------------------------------------

def _dist(vec1, vec2) -> Optional[float]:
    """
    Euclidean distance with None-masking, element-wise clipping to [-1, 1],
    and normalisation by sqrt(n_valid_dims).  Returns None if no valid dims.
    """
    a1 = np.array([x if x is not None else np.nan for x in vec1], dtype=float)
    a2 = np.array([x if x is not None else np.nan for x in vec2], dtype=float)
    valid = ~(np.isnan(a1) | np.isnan(a2))
    n = int(valid.sum())
    if n == 0:
        return None
    clipped = np.clip(a1[valid] - a2[valid], -1.0, 1.0)
    return float(np.linalg.norm(clipped) / math.sqrt(n))


def _sim(dist: float, activation: float, sensitivity: float, temperature: float = 1.0) -> float:
    """GCM/EBRW: exp((-sensitivity * dist + activation) / temperature)."""
    return math.exp((-sensitivity * dist + activation) / temperature)


def _norm(strengths: Dict) -> Dict:
    total = sum(strengths.values())
    if total < 1e-12:
        n = len(strengths)
        return {k: 1.0 / n for k in strengths} if n else {}
    return {k: v / total for k, v in strengths.items()}


def _uniform() -> Dict[int, float]:
    return {0: 0.5, 1: 0.5}


# ---------------------------------------------------------------------------
# Helpers shared by t-test focus selection
# ---------------------------------------------------------------------------

def _ttest_top_k(group0: np.ndarray, group1: np.ndarray, k: int) -> List[int]:
    """Return indices of the k features with the highest |t|-statistic."""
    n0, n1 = len(group0), len(group1)
    means0, means1 = np.nanmean(group0, axis=0), np.nanmean(group1, axis=0)
    vars0 = np.maximum(np.nanvar(group0, axis=0, ddof=1), 1e-8)
    vars1 = np.maximum(np.nanvar(group1, axis=0, ddof=1), 1e-8)
    se = np.sqrt(vars0 / n0 + vars1 / n1)
    se[se == 0] = 1e-12
    t = np.abs(means0 - means1) / se
    return np.argsort(t)[::-1][:k].tolist()


# ---------------------------------------------------------------------------
# SensitiveFeatures
# ---------------------------------------------------------------------------

class SensitiveFeatures(ReasoningStrategy):
    """Focus on k features that best discriminate classes via t-test."""

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="sensitive_features",
            display_name="Sensitive Features (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Focus on discriminative features identified via t-test",
            category="CoAX",
            parameters={
                "sensitivity":          {"default": 10.0,  "range": (1.0, 100.0)},
                "k":                    {"default": 3,      "range": (1, 10)},
                "decay_param":          {"default": 0.5,    "range": (0.1, 1.0)},
                "retrieval_threshold":  {"default": -2.5,   "range": (-5.0, 0.0)},
            },
        )

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity = float(extra.get("sensitivity", config.sensitivity))
        self.k = int(extra.get("k", 3))
        self._ct = 5.0          # constant time per trial (storage / feedback)
        self._ct_inf = 0.1      # constant time per inference

        self.memory = _MemoryBase(config.decay_param, config.retrieval_threshold)
        self.time: _Timer = config.time_manager if config.time_manager is not None else _Timer()

        self.focus_indices: List[int] = []
        self.last_probs: Optional[Dict] = None
        self.last_features = None
        self.last_explanation = None

    # -- focus selection --------------------------------------------------

    def _select_focus(self, features) -> None:
        """Set self.focus_indices via t-test; fall back to random shuffle."""
        n = len(features)
        k = max(1, self.k)

        def _random():
            idx = list(range(n))
            random.shuffle(idx)
            self.focus_indices = idx

        active = [ex for ex, _ in self.memory.retrieve(self.time.get_time())]
        if not active:
            return _random()

        groups: Dict = {}
        for ex in active:
            label = max(ex["label_probs"], key=ex["label_probs"].get)
            groups.setdefault(label, []).append(ex["features"])
        if len(groups) != 2:
            return _random()

        keys = list(groups.keys())
        g0, g1 = np.array(groups[keys[0]]), np.array(groups[keys[1]])
        if len(g0) <= 1 or len(g1) <= 1:
            return _random()

        self.focus_indices = _ttest_top_k(g0, g1, k)

    # -- lifecycle --------------------------------------------------------

    def new_instance(self) -> None:
        if self.last_probs is not None and self.last_features is not None:
            self.memory.store(self.last_probs, self.last_features,
                              self.last_explanation, self.time.get_time())
            self.time.add_time(self._ct)
        self.last_probs = self.last_features = self.last_explanation = None

    def infer(self, features, explanation=None, ai_prediction=None, **kwargs):
        t0 = self.time.get_time()
        self._select_focus(features)

        stored = self.memory.retrieve(self.time.get_time())
        self.time.add_time(self._ct_inf)

        self.last_features = features
        self.last_explanation = explanation

        if not stored:
            self.last_probs = _uniform()
            return self.last_probs, self.time.get_time() - t0, {}

        test = [features[i] if i in self.focus_indices else None for i in range(len(features))]
        strengths: Dict = {}
        for ex, act in stored:
            d = _dist(test, ex["features"])
            if d is None:
                continue
            s = _sim(d, act, self.sensitivity)
            for lbl, p in ex["label_probs"].items():
                strengths[lbl] = strengths.get(lbl, 0.0) + s * p

        self.last_probs = _norm(strengths) if strengths else _uniform()
        return self.last_probs, self.time.get_time() - t0, {"focus_features": self.focus_indices}

    def feedback(self, features, true_label: int, explanation=None, **kwargs) -> float:
        self.memory.store({true_label: 1.0}, features, explanation, self.time.get_time())
        self.last_probs = self.last_features = self.last_explanation = None
        self.time.add_time(self._ct)
        return self._ct

    def get_state(self) -> Dict:
        return {"memory_size": self.memory.size(), "focus_indices": self.focus_indices}


# ---------------------------------------------------------------------------
# SalientFeatures
# ---------------------------------------------------------------------------

class SalientFeatures(ReasoningStrategy):
    """Attend to top-k features by explanation value; re-mask stored exemplars."""

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="salient_features",
            display_name="Salient Features (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Focus on features with highest explanation value",
            category="CoAX",
            parameters={
                "sensitivity":          {"default": 10.0, "range": (1.0, 100.0)},
                "k":                    {"default": 3,    "range": (1, 10)},
                "decay_param":          {"default": 0.5,  "range": (0.1, 1.0)},
                "retrieval_threshold":  {"default": -2.5, "range": (-5.0, 0.0)},
            },
        )

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity = float(extra.get("sensitivity", config.sensitivity))
        self.k = int(extra.get("k", 3))
        self._ct = 5.0

        self.memory = _MemoryBase(config.decay_param, config.retrieval_threshold)
        self.time: _Timer = config.time_manager if config.time_manager is not None else _Timer()

        self.last_probs: Optional[Dict] = None
        self.last_features = None
        self.last_explanation = None

    def _mask(self, features, explanation):
        """Return feature list with only the top-k (by explanation value) non-None."""
        if explanation is None:
            return list(features)
        top_k = set(np.argsort(explanation)[-self.k:].tolist())
        return [features[i] if i in top_k else None for i in range(len(features))]

    def new_instance(self) -> None:
        if self.last_probs is not None and self.last_features is not None:
            masked = self._mask(self.last_features, self.last_explanation)
            self.memory.store(self.last_probs, masked,
                              self.last_explanation, self.time.get_time())
            self.time.add_time(self._ct)
        self.last_probs = self.last_features = self.last_explanation = None

    def infer(self, features, explanation=None, ai_prediction=None, **kwargs):
        t0 = self.time.get_time()
        self.last_explanation = explanation

        masked_query = self._mask(features, explanation)
        self.last_features = masked_query

        stored = self.memory.retrieve(self.time.get_time())
        self.time.add_time(self._ct)

        if not stored:
            self.last_probs = _uniform()
            return self.last_probs, self.time.get_time() - t0, {}

        query_vec = np.array(masked_query, dtype=object)
        strengths: Dict = {}
        valid = False

        for ex, act in stored:
            ex_exp = ex.get("explanation")
            if ex_exp is not None:
                ex_top_k = set(np.argsort(ex_exp)[-self.k:].tolist())
                ex_masked = [ex["features"][i] if i in ex_top_k else None
                             for i in range(len(ex["features"]))]
            else:
                ex_masked = ex["features"]

            d = _dist(query_vec, ex_masked)
            if d is None:
                continue
            valid = True
            s = _sim(d, act, self.sensitivity)
            for lbl, p in ex["label_probs"].items():
                strengths[lbl] = strengths.get(lbl, 0.0) + s * p

        self.last_probs = (_norm(strengths) if strengths else _uniform()) if valid else _uniform()
        return self.last_probs, self.time.get_time() - t0, {}

    def feedback(self, features, true_label: int, explanation=None, **kwargs) -> float:
        masked = self._mask(features, explanation)
        self.memory.store({true_label: 1.0}, masked, explanation, self.time.get_time())
        self.last_probs = self.last_features = self.last_explanation = None
        self.time.add_time(self._ct)
        return self._ct

    def get_state(self) -> Dict:
        return {"memory_size": self.memory.size()}


# ---------------------------------------------------------------------------
# ImportanceCategorization
# ---------------------------------------------------------------------------

class ImportanceCategorization(ReasoningStrategy):
    """Categorize via explanation vectors; use t-test to select focus dims."""

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="importance_categorization",
            display_name="Importance Categorization (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Categorize based on explanation dimensions using t-test",
            category="CoAX",
            parameters={
                "sensitivity":          {"default": 10.0, "range": (1.0, 100.0)},
                "k":                    {"default": 3,    "range": (1, 10)},
                "decay_param":          {"default": 0.5,  "range": (0.1, 1.0)},
                "retrieval_threshold":  {"default": -2.5, "range": (-5.0, 0.0)},
            },
        )

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity = float(extra.get("sensitivity", config.sensitivity))
        self.k = int(extra.get("k", 3))
        self._ct = 5.0

        self.memory = _MemoryBase(config.decay_param, config.retrieval_threshold)
        self.time: _Timer = config.time_manager if config.time_manager is not None else _Timer()

        self.last_probs: Optional[Dict] = None
        self.last_features = None
        self.last_explanation = None

    def _select_focus(self, explanation) -> List[int]:
        """Return focus indices from t-test; fall back to all indices."""
        n = len(explanation)
        active = [ex for ex, _ in self.memory.retrieve(self.time.get_time())
                  if ex.get("explanation") is not None]
        if not active:
            return list(range(n))

        groups: Dict = {}
        for ex in active:
            label = max(ex["label_probs"], key=ex["label_probs"].get)
            groups.setdefault(label, []).append(ex["explanation"])
        if len(groups) != 2:
            return list(range(n))

        keys = list(groups.keys())
        g0, g1 = np.array(groups[keys[0]]), np.array(groups[keys[1]])
        if len(g0) <= 1 or len(g1) <= 1:
            return list(range(n))

        return _ttest_top_k(g0, g1, max(1, self.k))

    def new_instance(self) -> None:
        if self.last_probs is not None and self.last_features is not None:
            self.memory.store(self.last_probs, self.last_features,
                              self.last_explanation, self.time.get_time())
            self.time.add_time(self._ct)
        self.last_probs = self.last_features = self.last_explanation = None

    def infer(self, features, explanation=None, ai_prediction=None, **kwargs):
        t0 = self.time.get_time()
        self.last_features = features
        self.last_explanation = explanation
        self.time.add_time(self._ct)          # advance time before retrieval (matches paper)

        if explanation is None:
            self.last_probs = _uniform()
            return self.last_probs, self.time.get_time() - t0, {}

        focus = self._select_focus(explanation)
        if not focus:
            self.last_probs = _uniform()
            return self.last_probs, self.time.get_time() - t0, {}

        stored = self.memory.retrieve(self.time.get_time())
        if not stored:
            self.last_probs = _uniform()
            return self.last_probs, self.time.get_time() - t0, {}

        test_vec = np.array([explanation[i] for i in focus], dtype=float)
        strengths: Dict = {}
        valid = False

        for ex, act in stored:
            if ex.get("explanation") is None:
                continue
            ex_vec = np.array([ex["explanation"][i] for i in focus], dtype=float)
            d = _dist(test_vec, ex_vec)
            if d is None:
                continue
            valid = True
            s = _sim(d, act, self.sensitivity)
            for lbl, p in ex["label_probs"].items():
                strengths[lbl] = strengths.get(lbl, 0.0) + s * p

        self.last_probs = _norm(strengths) if (valid and strengths) else _uniform()
        return self.last_probs, self.time.get_time() - t0, {"focus_dims": focus}

    def feedback(self, features, true_label: int, explanation=None, **kwargs) -> float:
        self.memory.store({true_label: 1.0}, features, explanation, self.time.get_time())
        self.last_probs = self.last_features = self.last_explanation = None
        self.time.add_time(self._ct)
        return self._ct

    def get_state(self) -> Dict:
        return {"memory_size": self.memory.size()}


# ---------------------------------------------------------------------------
# AttributionSum
# ---------------------------------------------------------------------------

class AttributionSum(ReasoningStrategy):
    """
    Two modes (explanation_type):
      'attribution' – bypass memory; sum top-k explanation values → logistic.
      'importance'  – memory-based per-feature importance with imputation.
    """

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="attribution_sum",
            display_name="Attribution Sum (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Sum top-k attributions; memory-based imputation when no explanation",
            category="CoAX",
            parameters={
                "sensitivity":          {"default": 15.0,         "range": (1.0, 100.0)},
                "scaling_factor":       {"default": 1.0,          "range": (0.1, 10.0)},
                "k":                    {"default": 2,             "range": (1, 10)},
                "decay_param":          {"default": 0.5,           "range": (0.1, 1.0)},
                "retrieval_threshold":  {"default": -0.3,          "range": (-5.0, 0.0)},
                "explanation_type":     {"default": "importance",  "options": ["importance", "attribution"]},
            },
        )

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity     = float(extra.get("sensitivity",    15.0))
        self.scaling_factor  = float(extra.get("scaling_factor", 1.0))
        self.k               = int(extra.get("k", 2))
        self.explanation_type = str(extra.get("explanation_type", "importance"))
        self._ct = 5.0

        self.memory = _MemoryBase(config.decay_param, config.retrieval_threshold)
        self.time: _Timer = config.time_manager if config.time_manager is not None else _Timer()

        self.last_probs: Optional[Dict] = None
        self.last_features = None
        self.last_explanation = None
        self.last_importances = None

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _logistic(x: float, scale: float) -> Tuple[float, float]:
        p1 = 1.0 / (1.0 + math.exp(-scale * x))
        return 1.0 - p1, p1

    # -- lifecycle --------------------------------------------------------

    def new_instance(self) -> None:
        if self.last_probs is not None and self.last_features is not None:
            t = self.time.get_time()
            if self.last_importances is not None:
                ranked = sorted(
                    enumerate(self.last_importances),
                    key=lambda iv: abs(iv[1]), reverse=True,
                )[:self.k]
            else:
                ranked = list(enumerate([None] * len(self.last_features)))

            for i, imp_val in ranked:
                feat_arr = [None] * len(self.last_features)
                feat_arr[i] = self.last_features[i]
                self.memory.store(self.last_probs, feat_arr, self.last_importances, t)

            self.time.add_time(self._ct)

        self.last_probs = self.last_features = self.last_explanation = None
        self.last_importances = None

    def infer(self, features, explanation=None, ai_prediction=None, **kwargs):
        t0 = self.time.get_time()
        self.last_features = features
        self.last_explanation = explanation
        self.last_importances = [0.0] * len(features) if features is not None else None

        # ── Case 1: attribution mode + explanation present ────────────────
        if self.explanation_type == "attribution" and explanation is not None:
            top_k = np.argsort(np.abs(explanation))[::-1][: self.k]
            total = sum(explanation[i] for i in top_k)
            if self.last_importances is not None:
                for i in range(len(explanation)):
                    self.last_importances[i] = explanation[i]
            p0, p1 = self._logistic(total, self.scaling_factor)
            self.last_probs = {0: p0, 1: p1}
            self.time.add_time(self._ct)
            return self.last_probs, self.time.get_time() - t0, {}

        # ── Case 2: memory-based ──────────────────────────────────────────
        total_attr = 0.0
        current_time = self.time.get_time()

        if explanation is not None and self.explanation_type == "importance":
            # Use top-k explanation dims; per-feature 1-D similarity in memory
            top_k = np.argsort(np.abs(explanation))[::-1][: self.k]
            for i in top_k:
                feat_val, exp_val = features[i], explanation[i]
                stored = self.memory.retrieve(current_time)
                local: Dict = {}
                for ex, act in stored:
                    if i < len(ex["features"]) and ex["features"][i] is not None:
                        d = abs(feat_val - ex["features"][i])
                        s = math.exp(-self.sensitivity * d + act)
                        for lbl, p in ex["label_probs"].items():
                            local[lbl] = local.get(lbl, 0.0) + s * p
                if not local:
                    continue
                local_norm = _norm(local)
                raw_sign = sum((1.0 if lbl == 1 else -1.0) * p for lbl, p in local_norm.items())
                if abs(raw_sign) < 0.01:
                    continue
                partial = (1.0 if raw_sign > 0 else -1.0) * exp_val
                total_attr += partial
                if self.last_importances is not None:
                    self.last_importances[i] = partial

        elif explanation is None and features is not None:
            # Impute importance by similarity-weighted averaging over all features
            stored = self.memory.retrieve(current_time)
            for i, feat_val in enumerate(features):
                sw_sum = 0.0
                sw_tot = 0.0
                for ex, act in stored:
                    if i < len(ex["features"]) and ex["features"][i] is not None:
                        d = abs(feat_val - ex["features"][i])
                        s = math.exp(-self.sensitivity * d + act)
                        label = max(ex["label_probs"], key=ex["label_probs"].get)
                        label_sign = 1.0 if label == 1 else -1.0
                        exp_vec = ex.get("explanation")
                        if exp_vec is None or i >= len(exp_vec):
                            continue
                        raw = exp_vec[i]
                        # flip sign if label and explanation disagree
                        ev = raw if math.copysign(1, raw) == label_sign else -raw
                        sw_sum += s * ev
                        sw_tot += s
                imputed = sw_sum / sw_tot if sw_tot > 0 else 0.0
                total_attr += imputed
                if self.last_importances is not None:
                    self.last_importances[i] = imputed

        total_attr = max(-1e3, min(1e3, total_attr))
        p0, p1 = self._logistic(total_attr, self.scaling_factor)
        self.last_probs = {0: p0, 1: p1}
        self.time.add_time(self._ct)
        return self.last_probs, self.time.get_time() - t0, {}

    def feedback(self, features, true_label: int, explanation=None, **kwargs) -> float:
        if self.explanation_type == "importance":
            global_lp: Optional[Dict] = {true_label: 1.0}
        else:
            global_lp = None

        current_time = self.time.get_time()

        if explanation is not None:
            top_k = np.argsort(np.abs(explanation))[::-1][: self.k]
            top_k_list = [(i, features[i], explanation[i]) for i in top_k]
        else:
            top_k_list = [(i, features[i], None) for i in range(len(features))]

        for i, val, exp_val in top_k_list:
            feat_arr = [None] * len(features)
            feat_arr[i] = val
            if self.explanation_type == "attribution":
                lp = {1: 1.0} if (exp_val is not None and exp_val >= 0) else {0: 1.0}
            else:
                lp = global_lp
            self.memory.store(lp, feat_arr, explanation, current_time)

        self.last_probs = self.last_features = self.last_explanation = None
        self.last_importances = None
        self.time.add_time(self._ct)
        return self._ct

    def get_state(self) -> Dict:
        return {"memory_size": self.memory.size(), "explanation_type": self.explanation_type}

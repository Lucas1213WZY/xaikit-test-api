"""Shared fit/inference contract for machine-proxy cognitive baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier


class DualInputClassifierBaseline:
    """Train feature-only and feature-plus-explanation proxy classifiers."""

    def __init__(self, *, smoothing_factor: float = 0.0) -> None:
        if smoothing_factor < 0:
            raise ValueError("smoothing_factor must be non-negative.")
        self.smoothing_factor = float(smoothing_factor)
        self._model: Any = None
        self._xai_model: Any = None
        self._feature_names: tuple[str, ...] | None = None
        self._explanation_names: tuple[str, ...] | None = None

    def fit(
        self,
        training_instances: Any,
        ai_predictions: Sequence[Any],
        *,
        explanations: Any = None,
    ) -> "DualInputClassifierBaseline":
        """Fit feature-only and, when supplied, feature-plus-XAI proxies."""
        x = self._prepare_fit_instances(training_instances)
        y = np.asarray(ai_predictions).reshape(-1)
        if len(x) == 0:
            raise ValueError("At least one training instance is required.")
        if len(x) != len(y):
            raise ValueError(
                "training_instances and ai_predictions must have the same length."
            )

        self._model = self._fit_estimator(x, y)
        self._xai_model = None
        self._explanation_names = None
        if explanations is not None:
            xai = self._prepare_fit_explanations(explanations)
            if len(xai) != len(x):
                raise ValueError(
                    "explanations and training_instances must have the same length."
                )
            if xai.shape[1] > 0:
                self._xai_model = self._fit_estimator(np.hstack([x, xai]), y)
        return self

    def predict(self, test_instances: Any, *, explanations: Any = None) -> np.ndarray:
        """Predict AI outputs for held-out instances without updating the proxy."""
        x, model = self._prepare_inference_input(test_instances, explanations)
        return np.asarray(model.predict(x))

    def predict_proba(
        self,
        test_instances: Any,
        *,
        explanations: Any = None,
    ) -> np.ndarray:
        """Return optionally smoothed class probabilities."""
        x, model = self._prepare_inference_input(test_instances, explanations)
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        if self.smoothing_factor:
            probabilities += self.smoothing_factor
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def score(
        self,
        test_instances: Any,
        ai_predictions: Sequence[Any],
        *,
        explanations: Any = None,
    ) -> float:
        """Return proxy fidelity: agreement with held-out AI predictions."""
        predictions = self.predict(test_instances, explanations=explanations)
        targets = np.asarray(ai_predictions).reshape(-1)
        if len(predictions) != len(targets):
            raise ValueError(
                "test_instances and ai_predictions must have the same length."
            )
        return float(np.mean(predictions == targets))

    def save(self, path: str | Path) -> Path:
        """Save both fitted proxy estimators."""
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DualInputClassifierBaseline":
        """Load a saved baseline of the requested class."""
        baseline = joblib.load(Path(path))
        if not isinstance(baseline, cls):
            raise TypeError(f"The saved object is not a {cls.__name__}.")
        return baseline

    def as_cognitive_model(
        self,
        cognitive_params: Mapping[str, float],
        dvs: Mapping[str, Sequence[Any]],
        trial_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Adapt the fitted proxy to the virtual-experiment executor."""
        attributes = trial_data.get("instance_attributes", {})
        explanation = {
            key: value
            for key, value in (
                trial_data.get("instance_explanation", {}) or {}
            ).items()
            if key.startswith("a") and key.endswith("_i")
        }
        visible_xai = explanation if self._trial_uses_xai(trial_data) else None
        prediction = self.predict(attributes, explanations=visible_xai)[0]
        confidence = float(
            np.max(self.predict_proba(attributes, explanations=visible_xai)[0])
        )
        prediction = prediction.item() if hasattr(prediction, "item") else prediction
        outputs: dict[str, Any] = {
            "agent_prediction": prediction,
            "ai_prediction": trial_data.get("ai_prediction"),
            "prob_correct": confidence,
            "pred_time": float(cognitive_params.get("pred_time", 0.0)),
        }
        for dv_name in dvs:
            outputs[dv_name] = (
                confidence
                if "prob" in dv_name.lower()
                else outputs["agent_prediction"]
            )
        return outputs

    def __call__(
        self,
        cognitive_params: Mapping[str, float],
        dvs: Mapping[str, Sequence[Any]],
        trial_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.as_cognitive_model(cognitive_params, dvs, trial_data)

    def _make_estimator(self) -> Any:
        raise NotImplementedError

    def _fit_estimator(self, x: np.ndarray, y: np.ndarray) -> Any:
        classes = np.unique(y)
        if len(classes) == 1:
            estimator = DummyClassifier(strategy="constant", constant=classes[0])
        else:
            estimator = self._make_estimator()
        estimator.fit(x, y)
        return estimator

    def _prepare_fit_instances(self, instances: Any) -> np.ndarray:
        if hasattr(instances, "columns") and hasattr(instances, "to_numpy"):
            self._feature_names = tuple(str(name) for name in instances.columns)
            instances = instances.to_numpy()
        else:
            self._feature_names = None
        return self._as_2d_numeric(np.asarray(instances))

    def _prepare_test_instances(self, instances: Any) -> np.ndarray:
        if isinstance(instances, Mapping):
            names = self._feature_names
            if names is None:
                values = list(instances.values())
            else:
                missing = [name for name in names if name not in instances]
                if missing:
                    raise ValueError(f"Test instance is missing features: {missing}.")
                values = [instances[name] for name in names]
            return self._as_2d_numeric(np.asarray(values))
        if hasattr(instances, "columns") and hasattr(instances, "to_numpy"):
            if self._feature_names is not None:
                missing = [
                    name for name in self._feature_names
                    if name not in instances.columns
                ]
                if missing:
                    raise ValueError(
                        f"Test instances are missing features: {missing}."
                    )
                instances = instances.loc[:, list(self._feature_names)]
            instances = instances.to_numpy()
        return self._as_2d_numeric(np.asarray(instances))

    def _prepare_fit_explanations(self, explanations: Any) -> np.ndarray:
        if hasattr(explanations, "columns") and hasattr(explanations, "to_numpy"):
            self._explanation_names = tuple(
                str(name) for name in explanations.columns
            )
            explanations = explanations.to_numpy()
        else:
            self._explanation_names = None
        return self._as_2d_numeric(np.asarray(explanations))

    def _prepare_test_explanations(self, explanations: Any) -> np.ndarray:
        if isinstance(explanations, Mapping):
            names = self._explanation_names
            if names is None:
                values = list(explanations.values())
            else:
                missing = [name for name in names if name not in explanations]
                if missing:
                    raise ValueError(
                        f"Test explanation is missing values: {missing}."
                    )
                values = [explanations[name] for name in names]
            return self._as_2d_numeric(np.asarray(values))
        if hasattr(explanations, "columns") and hasattr(
            explanations, "to_numpy"
        ):
            if self._explanation_names is not None:
                missing = [
                    name for name in self._explanation_names
                    if name not in explanations.columns
                ]
                if missing:
                    raise ValueError(
                        f"Test explanations are missing values: {missing}."
                    )
                explanations = explanations.loc[:, list(self._explanation_names)]
            explanations = explanations.to_numpy()
        return self._as_2d_numeric(np.asarray(explanations))

    def _prepare_inference_input(
        self,
        instances: Any,
        explanations: Any,
    ) -> tuple[np.ndarray, Any]:
        x = self._prepare_test_instances(instances)
        if explanations is None:
            return x, self._require_fitted()
        if self._xai_model is None:
            raise RuntimeError(
                "This baseline was not fitted with explanation vectors."
            )
        xai = self._prepare_test_explanations(explanations)
        if len(xai) != len(x):
            raise ValueError(
                "explanations and test_instances must have the same length."
            )
        return np.hstack([x, xai]), self._xai_model

    @staticmethod
    def _trial_uses_xai(trial_data: Mapping[str, Any]) -> bool:
        trial = trial_data.get("trial_info", {}) or {}
        method = str(
            trial.get("xai_method", trial.get("xai_type", "none"))
        ).lower()
        if method in {"none", "no_xai", "control"}:
            return False
        if str(trial.get("phase", "testing")).lower() == "training":
            return True
        tested_w_xai = trial.get("tested_w_xai", True)
        if isinstance(tested_w_xai, str):
            return tested_w_xai.strip().lower() in {"true", "1", "yes", "y"}
        return bool(tested_w_xai)

    @staticmethod
    def _as_2d_numeric(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("Instances must be a 2D array or table.")
        try:
            return values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Baseline instances and explanations must be numeric."
            ) from exc

    def _require_fitted(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                f"Call fit(...) before using the {self.__class__.__name__}."
            )
        return self._model

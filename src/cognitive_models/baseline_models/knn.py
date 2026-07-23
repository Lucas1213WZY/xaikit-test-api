"""A compact KNN baseline that learns to reproduce AI predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNBaseline:
    """KNN machine proxy trained on instances paired with AI predictions."""

    def __init__(self, n_neighbors: int = 5) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = int(n_neighbors)
        self._model: KNeighborsClassifier | None = None
        self._xai_model: KNeighborsClassifier | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._explanation_names: tuple[str, ...] | None = None

    def fit(
        self,
        training_instances: Any,
        ai_predictions: Sequence[Any],
        *,
        explanations: Any = None,
    ) -> "KNNBaseline":
        """Fit feature-only and, when supplied, feature-plus-XAI proxies."""
        x = self._prepare_fit_instances(training_instances)
        y = np.asarray(ai_predictions)
        if y.ndim != 1:
            y = y.reshape(-1)
        if len(x) == 0:
            raise ValueError("At least one training instance is required.")
        if len(x) != len(y):
            raise ValueError("training_instances and ai_predictions must have the same length.")

        effective_k = min(self.n_neighbors, len(x))
        self._model = KNeighborsClassifier(n_neighbors=effective_k)
        self._model.fit(x, y)

        self._xai_model = None
        self._explanation_names = None
        if explanations is not None:
            xai = self._prepare_fit_explanations(explanations)
            if xai.shape[1] > 0:
                if len(xai) != len(x):
                    raise ValueError(
                        "explanations and training_instances must have the same length."
                    )
                self._xai_model = KNeighborsClassifier(n_neighbors=effective_k)
                self._xai_model.fit(np.hstack([x, xai]), y)
        return self

    def predict(self, test_instances: Any, *, explanations: Any = None) -> np.ndarray:
        """Predict the AI output for unseen test instances without updating the model."""
        x, model = self._prepare_inference_input(test_instances, explanations)
        return model.predict(x)

    def predict_proba(self, test_instances: Any, *, explanations: Any = None) -> np.ndarray:
        """Return class probabilities for unseen test instances."""
        x, model = self._prepare_inference_input(test_instances, explanations)
        return model.predict_proba(x)

    def score(
        self,
        test_instances: Any,
        ai_predictions: Sequence[Any],
        *,
        explanations: Any = None,
    ) -> float:
        """Return proxy fidelity: agreement with held-out AI predictions."""
        x, model = self._prepare_inference_input(test_instances, explanations)
        return float(model.score(x, ai_predictions))

    def save(self, path: str | Path) -> Path:
        """Save the fitted baseline."""
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "KNNBaseline":
        """Load a saved baseline."""
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
        """Adapt the fitted baseline to ``xaikitTest.set_cognitive_model``."""
        attributes = trial_data.get("instance_attributes", {})
        explanation = {
            key: value
            for key, value in (trial_data.get("instance_explanation", {}) or {}).items()
            if key.startswith("a") and key.endswith("_i")
        }
        visible_explanation = explanation if self._trial_uses_xai(trial_data) else None
        prediction = self.predict(attributes, explanations=visible_explanation)[0]
        probabilities = self.predict_proba(
            attributes,
            explanations=visible_explanation,
        )[0]
        confidence = float(np.max(probabilities))
        ai_prediction = trial_data.get("ai_prediction")

        outputs: dict[str, Any] = {
            "agent_prediction": prediction.item() if hasattr(prediction, "item") else prediction,
            "ai_prediction": ai_prediction,
            "prob_correct": confidence,
            "pred_time": float(cognitive_params.get("pred_time", 0.0)),
        }
        for dv_name in dvs:
            key = dv_name.lower()
            outputs[dv_name] = confidence if "prob" in key else outputs["agent_prediction"]
        return outputs

    def __call__(
        self,
        cognitive_params: Mapping[str, float],
        dvs: Mapping[str, Sequence[Any]],
        trial_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Allow passing the fitted instance directly to ``set_cognitive_model``."""
        return self.as_cognitive_model(cognitive_params, dvs, trial_data)

    def _prepare_fit_instances(self, instances: Any) -> np.ndarray:
        if hasattr(instances, "columns") and hasattr(instances, "to_numpy"):
            self._feature_names = tuple(str(name) for name in instances.columns)
            array = instances.to_numpy()
        else:
            self._feature_names = None
            array = np.asarray(instances)
        return self._as_2d_numeric(array)

    def _prepare_test_instances(self, instances: Any) -> np.ndarray:
        if isinstance(instances, Mapping):
            if self._feature_names is None:
                values = list(instances.values())
            else:
                missing = [name for name in self._feature_names if name not in instances]
                if missing:
                    raise ValueError(f"Test instance is missing features: {missing}.")
                values = [instances[name] for name in self._feature_names]
            return self._as_2d_numeric(np.asarray(values))

        if hasattr(instances, "columns") and hasattr(instances, "to_numpy"):
            if self._feature_names is not None:
                missing = [name for name in self._feature_names if name not in instances.columns]
                if missing:
                    raise ValueError(f"Test instances are missing features: {missing}.")
                instances = instances.loc[:, list(self._feature_names)]
            instances = instances.to_numpy()
        return self._as_2d_numeric(np.asarray(instances))

    def _prepare_fit_explanations(self, explanations: Any) -> np.ndarray:
        if hasattr(explanations, "columns") and hasattr(explanations, "to_numpy"):
            self._explanation_names = tuple(str(name) for name in explanations.columns)
            array = explanations.to_numpy()
        else:
            self._explanation_names = None
            array = np.asarray(explanations)
        return self._as_2d_numeric(array)

    def _prepare_test_explanations(self, explanations: Any) -> np.ndarray:
        if isinstance(explanations, Mapping):
            if self._explanation_names is None:
                values = list(explanations.values())
            else:
                missing = [
                    name for name in self._explanation_names
                    if name not in explanations
                ]
                if missing:
                    raise ValueError(f"Test explanation is missing values: {missing}.")
                values = [explanations[name] for name in self._explanation_names]
            return self._as_2d_numeric(np.asarray(values))

        if hasattr(explanations, "columns") and hasattr(explanations, "to_numpy"):
            if self._explanation_names is not None:
                missing = [
                    name for name in self._explanation_names
                    if name not in explanations.columns
                ]
                if missing:
                    raise ValueError(f"Test explanations are missing values: {missing}.")
                explanations = explanations.loc[:, list(self._explanation_names)]
            explanations = explanations.to_numpy()
        return self._as_2d_numeric(np.asarray(explanations))

    def _prepare_inference_input(
        self,
        instances: Any,
        explanations: Any,
    ) -> tuple[np.ndarray, KNeighborsClassifier]:
        x = self._prepare_test_instances(instances)
        if explanations is None:
            return x, self._require_fitted()
        if self._xai_model is None:
            raise RuntimeError(
                "This baseline was not fitted with explanation vectors."
            )
        xai = self._prepare_test_explanations(explanations)
        if len(xai) != len(x):
            raise ValueError("explanations and test_instances must have the same length.")
        return np.hstack([x, xai]), self._xai_model

    @staticmethod
    def _trial_uses_xai(trial_data: Mapping[str, Any]) -> bool:
        trial = trial_data.get("trial_info", {}) or {}
        method = str(trial.get("xai_method", trial.get("xai_type", "none"))).lower()
        if method in {"none", "no_xai", "control"}:
            return False
        if str(trial.get("phase", "testing")).lower() == "training":
            return True
        tested_w_xai = trial.get("tested_w_xai", True)
        if isinstance(tested_w_xai, str):
            return tested_w_xai.strip().lower() in {"true", "1", "yes", "y"}
        return bool(tested_w_xai)

    @staticmethod
    def _as_2d_numeric(instances: np.ndarray) -> np.ndarray:
        if instances.ndim == 1:
            instances = instances.reshape(1, -1)
        if instances.ndim != 2:
            raise ValueError("Instances must be a 2D array or table.")
        try:
            return instances.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("KNN instances must contain only numeric features.") from exc

    def _require_fitted(self) -> KNeighborsClassifier:
        if self._model is None:
            raise RuntimeError("Call fit(...) before using the KNN baseline.")
        return self._model

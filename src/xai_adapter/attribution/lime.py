"""LIME-backed tabular attribution methods."""

from __future__ import annotations

from typing import Callable, List, Literal, Optional

import numpy as np

from ..base import ArrayLike, PostprocessFn, PreprocessFn, XAIAdapterResult, ensure_2d
from .base import LocalAttribution, _extract_interpret_scores


class Lime(LocalAttribution):
    """LIME tabular method with a sklearn-like fit/explain API.

    Parameters
    ----------
    backend : 'native' | 'interpret'
        'native'    — uses lime.lime_tabular.LimeTabularExplainer directly
                      (default, no extra dependency).
        'interpret' — uses interpret.blackbox.LimeTabular, which wraps the
                      same LIME library but returns an InterpretML Explanation
                      object stored in metadata['raw_explanation'] so you can
                      call show(result.metadata['raw_explanation']) for the
                      InterpretML dashboard.  Algorithm is identical.
    """

    method_name = "lime"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_data: Optional[ArrayLike] = None,
        training_labels: Optional[ArrayLike] = None,
        categorical_features: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        kernel_width: float = 1.5,
        num_samples: int = 5000,
        n_bins: int = 4,
        target: int = 1,
        backend: Literal['native', 'interpret'] = 'native',
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.predict_fn = predict_fn
        self.training_labels = training_labels
        self.categorical_features = categorical_features or []
        self.feature_names = feature_names
        self.kernel_width = float(kernel_width)
        self.num_samples = int(num_samples)
        self.n_bins = int(n_bins)
        self.backend = backend
        self.explainer = None
        if training_data is not None:
            self.fit(training_data, y=training_labels)

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        """Fit the LIME tabular explainer on training data."""
        training_data = ensure_2d(X)
        training_labels = y if y is not None else self.training_labels
        self.feature_names = self.feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]

        if self.backend == 'interpret':
            try:
                from interpret.blackbox import LimeTabular
            except ImportError as exc:
                raise ImportError("InterpretML is required for backend='interpret'. "
                                  "Install with: pip install interpret") from exc
            self.explainer = LimeTabular(
                predict_fn=lambda x: self.predict_fn(self.preprocessing_fn(x)),
                data=training_data,
                feature_names=self.feature_names,
            )
        else:
            try:
                import lime.lime_tabular as lime_tabular
                from lime.discretize import BaseDiscretizer
            except ImportError as exc:
                raise ImportError("LIME is required. Install with: pip install lime") from exc

            class PercentileDiscretizer(BaseDiscretizer):
                def __init__(self, data, categorical_features, feature_names,
                             labels=None, random_state=None, bins=4):
                    self.num_bins = int(bins)
                    super().__init__(data, categorical_features, feature_names,
                                    labels=labels, random_state=random_state)

                def bins(self, data, labels):
                    return [
                        np.percentile(data[:, f], np.linspace(0, 100, self.num_bins + 1))
                        for f in self.to_discretize
                    ]

            discretizer = PercentileDiscretizer(
                training_data,
                categorical_features=self.categorical_features,
                feature_names=self.feature_names,
                labels=training_labels,
                bins=self.n_bins,
            )
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                mode="classification",
                training_labels=training_labels,
                categorical_features=self.categorical_features,
                feature_names=self.feature_names,
                feature_selection="auto",
                kernel_width=self.kernel_width,
                discretizer=discretizer,
                discretize_continuous=True,
                sample_around_instance=True,
            )

        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)

        if self.backend == 'interpret':
            return self._explain_interpret(raw_instances)
        return self._explain_native(raw_instances)

    def _explain_native(self, raw_instances: np.ndarray) -> XAIAdapterResult:
        values, base_values, explanation_objects = [], [], []
        for instance in raw_instances:
            exp = self.explainer.explain_instance(
                instance,
                lambda x: self.predict_fn(self.preprocessing_fn(x)),
                num_features=min(50, len(self.feature_names)),
                num_samples=self.num_samples,
            )
            row = np.zeros(len(self.feature_names), dtype=float)
            for feat_idx, importance in dict(exp.as_map().get(self.target, {})).items():
                row[feat_idx] = importance
            values.append(row)
            base_values.append(float(exp.intercept.get(self.target, 0.0)))
            explanation_objects.append(exp)

        return XAIAdapterResult(
            values=self._postprocess_values(raw_instances, np.asarray(values)),
            base_values=np.asarray(base_values, dtype=float),
            method=self.method_name,
            metadata={"explanation_objects": explanation_objects, "backend": "native"},
        )

    def _explain_interpret(self, raw_instances: np.ndarray) -> XAIAdapterResult:
        n_features = len(self.feature_names)
        # interpret.blackbox.LimeTabular requires y (uses it for error colouring)
        y_pred = np.argmax(self.predict_fn(self.preprocessing_fn(raw_instances)), axis=1)
        explanation = self.explainer.explain_local(raw_instances, y_pred)
        values, base_values = _extract_interpret_scores(explanation, raw_instances.shape[0], n_features)
        return XAIAdapterResult(
            values=self._postprocess_values(raw_instances, values),
            base_values=base_values,
            method=self.method_name,
            metadata={"raw_explanation": explanation, "backend": "interpret"},
        )


__all__ = ["Lime"]
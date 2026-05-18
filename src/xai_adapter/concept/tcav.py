"""TCAV (Testing with Concept Activation Vectors) adapter via Captum."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..base import (
    ArrayLike,
    XAIAdapter,
    XAIAdapterResult,
    ensure_2d,
)


class TCAVAdapter(XAIAdapter):
    """TCAV adapter using captum.concept.TCAV.

    TCAV measures a model's conceptual sensitivity: for each concept
    (described by positive/negative example sets), it computes a
    directional derivative that quantifies how strongly the concept
    influences the model's output for a class.

    The ``explain()`` method returns an ``XAIAdapterResult`` where:
      - ``values`` has shape ``(n_instances, n_concepts)`` — per-instance
        TCAV scores for each concept averaged across all trained CAVs
        for that concept.
      - ``base_values`` is zeros (TCAV has no baseline).
      - ``metadata["concept_names"]`` lists concept names in column order.
      - ``metadata["tcav_scores"]`` holds the raw captum TCAV output dict.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to explain.
    layer : str or torch.nn.Module
        Layer name (or module) at which to compute directional derivatives.
    concepts : list of dict
        Each dict has keys:
          ``"name"``     — concept label string
          ``"positive"`` — array-like of positive example tensors
          ``"negative"`` — array-like of negative example tensors
    classifier : captum.concept.Classifier, optional
        Custom classifier for training CAVs (default: linear SVM via
        captum's default Classifier).
    n_trials : int
        Number of random splits for statistical robustness (default 1).
    target : int
        Output class index (default 1).
    device : str
        PyTorch device (default 'cpu').
    save_path : str or Path, optional
        Directory where CAV checkpoints are saved.
    """

    method_name = "tcav"

    def __init__(
        self,
        *,
        model: Any,
        layer: Any,
        concepts: List[Dict],
        classifier: Any = None,
        n_trials: int = 1,
        target: int = 1,
        device: str = "cpu",
        save_path: Optional[Union[str, Path]] = None,
        preprocessing_fn: Optional[Callable] = None,
    ):
        super().__init__(target=target)
        try:
            import torch
            from captum.concept import TCAV, Concept
        except ImportError as exc:
            raise ImportError(
                "PyTorch and Captum are required for TCAVAdapter. "
                "Install with: pip install torch captum"
            ) from exc

        self.torch = torch
        self.model = model
        self.device = device
        self.n_trials = n_trials
        self.preprocessing_fn = preprocessing_fn or (lambda x: x)

        self.model.eval()
        self.model.to(device)

        self._concept_defs = concepts
        self._concept_names = [c["name"] for c in concepts]

        self._save_path = str(save_path) if save_path else ".tcav_cavs"
        self._classifier = classifier

        tcav_kwargs = {}
        if classifier is not None:
            tcav_kwargs["classifier"] = classifier

        self.tcav = TCAV(
            model=self.model,
            layers=[layer] if not isinstance(layer, list) else layer,
            save_path=self._save_path,
            **tcav_kwargs,
        )
        self._captum_concepts = self._build_concepts(concepts)
        self.is_fitted = True

    def _build_concepts(self, concepts: List[Dict]) -> List[Any]:
        """Convert concept dicts to captum.concept.Concept objects."""
        from captum.concept import Concept
        from torch.utils.data import DataLoader, TensorDataset

        built = []
        for idx, c in enumerate(concepts):
            pos = self.torch.tensor(
                ensure_2d(np.asarray(c["positive"])),
                dtype=self.torch.float32,
                device=self.device,
            )
            loader = DataLoader(TensorDataset(pos), batch_size=max(1, len(pos)))
            built.append(Concept(id=idx, name=c["name"], data_iter=loader))
        return built

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """Compute TCAV scores for each instance against each concept.

        Returns
        -------
        XAIAdapterResult
            values  : (n_instances, n_concepts) TCAV sensitivity scores
            metadata: concept_names, raw tcav_scores dict from Captum
        """
        self._require_fitted()
        raw = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, device=self.device)

        experimental_sets = [self._captum_concepts]

        scores_raw = self.tcav.interpret(
            inputs=x,
            experimental_sets=experimental_sets,
            target=self.target,
            n_steps=self.n_trials,
        )

        n_instances = x.shape[0]
        n_concepts = len(self._concept_names)
        values = np.zeros((n_instances, n_concepts), dtype=float)

        for col_idx, concept in enumerate(self._captum_concepts):
            key = concept.name
            if key in scores_raw:
                layer_scores = scores_raw[key]
                layer_vals = list(layer_scores.values())
                if layer_vals:
                    tcav_score = float(np.mean([v.item() if hasattr(v, "item") else float(v)
                                                for v in layer_vals]))
                    values[:, col_idx] = tcav_score

        return XAIAdapterResult(
            values=values,
            base_values=np.zeros(n_instances, dtype=float),
            method=self.method_name,
            metadata={
                "concept_names": self._concept_names,
                "tcav_scores": scores_raw,
            },
        )


__all__ = ["TCAVAdapter"]

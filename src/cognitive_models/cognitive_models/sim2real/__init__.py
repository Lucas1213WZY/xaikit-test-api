"""Sim2Real cognitive strategies.

Three models of how a participant answers the counterfactual question "does the
AI's prediction go up or down?". All three share the same two-stage shape --
score the original instance, score the counterfactual, then push the difference
through a learned logistic -- and differ only in what they read as evidence:

* :class:`~.gcm_strategies.Sim2RealFittedAttributionSum` sums the explanation's
  attributions. It is the strongest model wherever the changed feature has a
  visible attribution, and answers at chance wherever it does not.
* :class:`~.gcm_strategies.Sim2RealFittedSensitiveFeatures` ignores the
  explanation entirely and classifies by exemplar similarity over feature
  *values*, which is CoAX's ``none``-condition strategy.
* :class:`~.gcm_strategies.Sim2RealFittedSalientFeatures` uses the explanation
  only to aim attention, then also reasons over values.

Fitted per participant, the split is clean: the attribution model wins every
participant in ``faithful`` and ``robust``, and the value-based strategies win
the majority in ``sparse`` and ``sparse_robust`` -- the two conditions where the
changed feature is usually invisible.
"""

from .gcm_strategies import (
    AttributionSumComparison,
    ProjectedAttributionPair,
    Sim2RealAttributionProjector,
    Sim2RealFittedAttributionSum,
)
from .gcm_strategies import (
    Sim2RealFittedSalientFeatures,
    Sim2RealFittedSensitiveFeatures,
    Sim2RealValueEncoder,
)

__all__ = [
    "AttributionSumComparison",
    "ProjectedAttributionPair",
    "Sim2RealAttributionProjector",
    "Sim2RealFittedAttributionSum",
    "Sim2RealFittedSalientFeatures",
    "Sim2RealFittedSensitiveFeatures",
    "Sim2RealValueEncoder",
]

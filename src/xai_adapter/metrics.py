"""Property metrics for feature-attribution explanations.

Two generations of metric live here.

The **original three** -- a reconstruction loss, a non-zero count, and a local
Lipschitz estimate -- were written for ``Sim2RealPropertyAttribution``, which
uses them *generatively*: it random-searches candidate attribution matrices and
keeps whichever minimises the loss. That caller passes synthetic models whose
``predict`` returns 1-D labels and whose weights are true linear coefficients,
so its defaults are preserved exactly (``convention="coefficient"``,
``radius=1.0``, ``empty_neighborhood="zero"``).

The **standard set** was added for evaluating real explanations. Quantus
(JMLR 2023) groups tabular XAI metrics into faithfulness, complexity and
robustness; this module now covers all three:

* faithfulness -- deletion-curve AOPC and Faithfulness Correlation
  (Bhatt et al. 2020). Both are *perturbation-based*: mask the features an
  explanation calls important and check the model's output actually moves.
* complexity -- effective complexity (non-zero count), Gini sparseness
  (Chalasani et al. 2020) and attribution entropy (Bhatt et al. 2020).
* robustness -- the local Lipschitz estimate of Alvarez-Melis & Jaakkola (2018)
  and Relative Input Stability (Agarwal et al. 2022). The first is scale
  dependent and so compares runs of one method; the second normalises both
  differences by their own magnitude and so compares across methods.

Why perturbation-based faithfulness rather than the reconstruction loss: the
reconstruction loss has to know whether a method's numbers are *coefficients*
(multiply by the feature value) or *contributions* (already the effect), and in
which space the sum lives (probability, margin, logit). Get that wrong and it
returns a plausible number rather than an error. Perturbation metrics never ask
-- they only need the attribution *ranking* -- so they work unchanged across
LIME, SHAP, and the decision-tree/logistic-regression surrogates.

The reconstruction loss is kept, now convention-aware, because it is exact and
free for genuinely additive methods, and because sim2real's search depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


#: Columns the explanation stage attaches to each explanation row. Deliberately
#: none of these match ``startswith("a") and endswith("_i")``, the pattern four
#: separate call sites use to find attribution columns (``src/api.py``,
#: ``virtual_experiment_executor/executor.py``, and the two CoAX modules) -- a
#: quality column mistaken for an attribution would be fed to a cognitive model
#: as if it were feature evidence.
QUALITY_COLUMNS: tuple[str, ...] = (
    "faithfulness_aopc",
    "faithfulness_corr",
    "faithfulness_loss",
    "sparsity_nonzero",
    "sparsity_gini",
    "complexity_entropy",
    "robustness_lipschitz",
    "robustness_ris",
    "quality_note",
)

#: Which quality columns are numeric, in the order a summary table shows them.
QUALITY_METRIC_COLUMNS: tuple[str, ...] = QUALITY_COLUMNS[:-1]

#: Higher-is-better metrics, for the optional score view.
_HIGHER_IS_BETTER = frozenset({"faithfulness_aopc", "faithfulness_corr", "sparsity_gini"})


def ensure_2d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


# -- reading a model's output --------------------------------------------


def model_scores(
    model_or_fn,
    X,
    *,
    target: Optional[int] = 1,
) -> tuple[np.ndarray, Optional[np.ndarray], str]:
    """Labels, target-class scores, and which space the scores live in.

    Models in this repository return *scores*, not labels: the torch MLP returns
    ``(n, 2)`` softmax rows and the Keras one returns probabilities (see the note
    at ``coxam_study_runner._label_from_predictions``). Flattening that with
    ``reshape(-1)`` yields ``2n`` values, which then silently mis-compares
    against ``n`` reconstructions. So conversion goes through the same helper
    ``classification_metrics`` uses.

    Returns ``(labels, target_scores, space)`` where ``space`` is one of
    ``"probability"``, ``"margin"``, ``"label"`` or ``"multiclass"``. A caller
    that cannot interpret ``"multiclass"`` should decline to score rather than
    guess which column is the positive class.
    """
    from src.ai_models.model_api import _labels_and_scores_from_predictions

    X_arr = ensure_2d(X)
    if callable(model_or_fn) and not hasattr(model_or_fn, "predict"):
        raw = np.asarray(model_or_fn(X_arr))
    else:
        raw = np.asarray(model_or_fn.predict(X_arr))

    labels, scores = _labels_and_scores_from_predictions(raw)
    labels = np.asarray(labels).reshape(-1)

    if raw.ndim == 2 and raw.shape[1] > 2:
        return labels, None, "multiclass"
    if scores is None:
        return labels, None, "label"

    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 2:
        column = 1 if target is None else int(target)
        column = min(max(column, 0), scores_arr.shape[1] - 1)
        return labels, scores_arr[:, column].astype(float), "probability"

    flat = scores_arr.reshape(-1).astype(float)
    space = "probability" if (flat.min() >= 0.0 and flat.max() <= 1.0) else "margin"
    return labels, flat, space


def _predict(model_or_fn, X: np.ndarray) -> np.ndarray:
    """Backwards-compatible raw call, kept for callers that want the raw output."""
    if callable(model_or_fn) and not hasattr(model_or_fn, "predict"):
        return np.asarray(model_or_fn(X))
    return np.asarray(model_or_fn.predict(X))


# -- how a method's numbers must be read ---------------------------------


@dataclass(frozen=True)
class QualitySpec:
    """How one XAI method's numbers must be read before they can be scored.

    ``convention``
        ``"contribution"`` -- the value already *is* the effect (SHAP, LIME,
        integrated gradients); reconstruct with ``base + sum(w)``.
        ``"coefficient"`` -- the value multiplies the feature (linear weights,
        sim2real ground truth); reconstruct with ``base + sum(x * w)``.
        ``"indicator"`` -- the value marks membership rather than magnitude
        (decision-tree paths, rule sets); not additive, so no reconstruction.
    ``link``
        ``"identity"`` when the reconstruction lands in the model's own output
        space, ``"logit"`` when it lands in log-odds. The logistic-regression
        surrogate returns ``coef x value`` contributions in **logit** space while
        the AI model returns softmax probabilities -- comparing them directly
        would be a units error.
    ``fidelity``
        ``"score"`` compares reconstructed against the model's score;
        ``"label"`` compares implied class against the model's label;
        ``"none"`` means the reconstruction is meaningless for this method.
    """

    convention: str = "contribution"
    link: str = "identity"
    fidelity: str = "score"
    note: str = ""


_NOT_ADDITIVE = "path indicators are not additive; see the method's own fidelity"

#: Per-method reading conventions, keyed by the names in ``registry.py``.
#: Mutable on purpose -- ``register_quality_spec`` lets a method defined
#: outside this package declare how its numbers must be read.
DEFAULT_QUALITY_SPECS: dict[str, QualitySpec] = {
    # Additive attributions in the model's own output space.
    **{
        name: QualitySpec("contribution", "identity", "score")
        for name in (
            "lime", "lime_tabular",
            "shap", "shap_kernel", "shap_tree", "shap_linear", "shap_deep", "shap_gradient",
            "integrated_gradients", "ig", "deeplift",
            "gradient_input", "gradient_x_input", "input_gradients", "lrp",
            "ebm", "lofo", "leave_one_feature_out",
        )
    },
    # coef x value contributions, but summed in logit space.
    "logistic_regression": QualitySpec("contribution", "logit", "label"),
    "lr": QualitySpec("contribution", "logit", "label"),
    "weights": QualitySpec("contribution", "logit", "label"),
    # True linear coefficients against a synthetic function.
    "sim2real_property": QualitySpec("coefficient", "identity", "label"),
    # Structure, not magnitude: rank and stability still mean something, the
    # reconstruction does not.
    **{
        name: QualitySpec("indicator", "identity", "none", _NOT_ADDITIVE)
        for name in (
            "decision_tree", "dt", "rules", "rule_list", "rule_set", "anchors",
            "sklearn_global", "global_feature_importance", "tcav",
            "counterfactual", "cf", "wachter", "dice", "prototypes",
        )
    },
}


def register_quality_spec(method_key: str, spec: QualitySpec) -> None:
    """Declare how a method's attribution values must be read.

    Any adapter can be scored without this -- the perturbation, complexity and
    robustness metrics need only the attribution *ranking*, never its units.
    Registering additionally unlocks the exact reconstruction loss, which does
    need to know whether the values are coefficients or contributions.

    Mirrors ``register_xai_method`` in ``registry.py``, so a method defined
    outside this package can opt in:

        register_quality_spec("my_method", QualitySpec("contribution", "identity", "score"))
    """
    if not isinstance(spec, QualitySpec):
        raise TypeError(f"spec must be a QualitySpec, got {type(spec).__name__}.")
    DEFAULT_QUALITY_SPECS[str(method_key).strip().lower()] = spec


def _resolve_method_key(method_key: str) -> Optional[str]:
    """Match a result's method name against the declared specs.

    Adapters name their results in several shapes -- a bare registry key
    (``"lime"``), a variant suffix (``"sim2real_property_faithful"``), or a
    target suffix. So an exact match is tried first, then the longest declared
    key the name starts with.
    """
    key = str(method_key or "").strip().lower()
    if not key:
        return None
    if key in DEFAULT_QUALITY_SPECS:
        return key
    candidates = [name for name in DEFAULT_QUALITY_SPECS if key.startswith(name)]
    return max(candidates, key=len) if candidates else None


def quality_spec(
    method_key: str,
    *,
    overrides: Optional[Mapping[str, QualitySpec]] = None,
) -> QualitySpec:
    """The reading convention for one method, or a conservative default.

    An unknown method gets ``fidelity="none"``: scoring its reconstruction would
    require guessing whether its numbers are coefficients or contributions, and
    a wrong guess produces a plausible number rather than an error. It still
    receives every convention-free metric.
    """
    key = str(method_key or "").strip().lower()
    if overrides:
        if key in overrides:
            return overrides[key]
        prefix = [name for name in overrides if key.startswith(name)]
        if prefix:
            return overrides[max(prefix, key=len)]
    resolved = _resolve_method_key(key)
    if resolved is not None:
        return DEFAULT_QUALITY_SPECS[resolved]
    return QualitySpec(
        "contribution", "identity", "none",
        f"unknown method {method_key!r}: attribution convention not declared",
    )


# -- faithfulness: reconstruction (exact, additive methods only) ----------


def reconstructed_scores(
    X,
    W,
    base_values: Optional[np.ndarray] = None,
    *,
    convention: str = "coefficient",
    link: str = "identity",
) -> np.ndarray:
    """What the explanation says the model's output is, per instance."""
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    base = (
        np.zeros(X_arr.shape[0], dtype=float)
        if base_values is None
        else np.asarray(base_values, dtype=float).reshape(-1)
    )
    if convention == "coefficient":
        total = base + np.sum(X_arr * W_arr, axis=1)
    elif convention == "contribution":
        total = base + np.sum(W_arr, axis=1)
    else:
        raise ValueError(
            f"Cannot reconstruct from convention {convention!r}; "
            "only 'coefficient' and 'contribution' are additive."
        )
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-total))
    return total


def _reconstruction_threshold(
    convention: str, link: str, space: str, explicit: Optional[float]
) -> float:
    """Where the reconstructed value crosses from class 0 to class 1.

    A coefficient reconstruction is a margin (``base + x.w > 0``), which is what
    the sim2real search depends on. A contribution reconstruction lands wherever
    the model's own output lives -- 0.5 for probabilities. Getting this wrong is
    not subtle: on probability outputs a ``> 0`` test calls every instance
    class 1.
    """
    if explicit is not None:
        return float(explicit)
    if link == "logit":
        return 0.5
    if convention == "contribution" and space == "probability":
        return 0.5
    return 0.0


def fidelity_regression_loss(
    model_or_fn,
    X,
    W,
    *,
    base_values: Optional[np.ndarray] = None,
    convention: str = "coefficient",
    link: str = "identity",
    target: Optional[int] = 1,
) -> np.ndarray:
    """Squared reconstruction error between the model's score and the explanation's."""
    X_arr = ensure_2d(X)
    reconstructed = reconstructed_scores(
        X_arr, W, base_values, convention=convention, link=link
    )
    _labels, scores, _space = model_scores(model_or_fn, X_arr, target=target)
    if scores is None:
        true_values = _predict(model_or_fn, X_arr).reshape(-1).astype(float)
    else:
        true_values = scores
    return (true_values - reconstructed) ** 2


def fidelity_classification_loss(
    model_or_fn,
    X,
    W,
    *,
    base_values: Optional[np.ndarray] = None,
    convention: str = "coefficient",
    link: str = "identity",
    target: Optional[int] = 1,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """0/1 disagreement between the model's label and the explanation's implied label."""
    X_arr = ensure_2d(X)
    reconstructed = reconstructed_scores(
        X_arr, W, base_values, convention=convention, link=link
    )
    labels, _scores, space = model_scores(model_or_fn, X_arr, target=target)
    cut = _reconstruction_threshold(convention, link, space, threshold)
    implied = (reconstructed > cut).astype(int)
    return (labels.astype(int) != implied).astype(float)


# -- faithfulness: perturbation-based (the standard) ----------------------



def class_scores(model_or_fn, X, labels: np.ndarray) -> Optional[np.ndarray]:
    """The model's score for one nominated class per instance.

    Perturbation metrics measure how far the model's confidence *in what it
    predicted* falls when evidence is removed. Scoring a fixed class instead
    makes the metric cancel itself out on any dataset whose predictions are
    mixed: on adult, masking the top features lowers p(class 1) for the 70
    instances predicted 1 (+0.43) and raises it for the 230 predicted 0
    (-0.14), averaging to roughly zero while both halves behaved exactly as a
    faithful explanation should.
    """
    X_arr = ensure_2d(X)
    raw = _predict(model_or_fn, X_arr)
    rows = np.arange(X_arr.shape[0])
    idx = np.asarray(labels).reshape(-1).astype(int)

    if raw.ndim == 2:
        if raw.shape[1] <= idx.max():
            return None
        return raw[rows, idx].astype(float)

    flat = raw.reshape(-1).astype(float)
    if flat.min() < 0.0 or flat.max() > 1.0:
        return None  # a margin, not a probability: no complement to take
    return np.where(idx == 1, flat, 1.0 - flat)


def _masked(X: np.ndarray, baseline: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``X`` with the given per-row column indices replaced by the baseline."""
    out = X.copy()
    rows = np.arange(X.shape[0])[:, None]
    out[rows, columns] = baseline[columns]
    return out


def deletion_curve(
    model_or_fn,
    X,
    W,
    *,
    baseline,
    target: Optional[int] = 1,
    steps: Optional[int] = None,
    against: str = "predicted",
) -> np.ndarray:
    """Score drop as the most-attributed features are progressively masked.

    Returns ``(n_instances, steps)``: entry ``[i, k]`` is
    ``f(x_i) - f(x_i with its top k+1 features replaced by the baseline)``.
    A faithful explanation ranks features so that masking them costs the model
    the most, so a larger drop is better.

    Cost is ``steps + 1`` batched ``predict`` calls regardless of instance
    count. These datasets carry 5-12 raw features and the explanation stage caps
    instances at 300, so this is a dozen calls over 300 rows.
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    base_row = np.asarray(baseline, dtype=float).reshape(-1)
    n_features = X_arr.shape[1]
    steps = n_features if steps is None else int(min(steps, n_features))

    order = np.argsort(-np.abs(W_arr), axis=1)
    labels, target_scores, _space = model_scores(model_or_fn, X_arr, target=target)

    # Which class the confidence is measured on, held fixed across mask sizes so
    # every step compares like with like.
    if against == "predicted":
        original = class_scores(model_or_fn, X_arr, labels)
        scored_class = labels
    else:
        original, scored_class = target_scores, None
    if original is None:
        return np.full((X_arr.shape[0], steps), np.nan)

    drops = np.empty((X_arr.shape[0], steps), dtype=float)
    for k in range(steps):
        masked = _masked(X_arr, base_row, order[:, : k + 1])
        if scored_class is None:
            _l, scores, _s = model_scores(model_or_fn, masked, target=target)
        else:
            scores = class_scores(model_or_fn, masked, scored_class)
        drops[:, k] = original - (scores if scores is not None else original)
    return drops


def faithfulness_aopc(
    model_or_fn,
    X,
    W,
    *,
    baseline,
    target: Optional[int] = 1,
    steps: Optional[int] = None,
    against: str = "predicted",
) -> np.ndarray:
    """Area over the deletion curve: mean score drop across mask sizes.

    Higher is better. This is the headline faithfulness number because it needs
    only the attribution *ranking* -- no convention, no link function -- so it is
    comparable across LIME, SHAP and the surrogates.

    ``against="predicted"`` (the default) measures the fall in the model's
    confidence in *what it predicted*, which is what the pixel-flipping
    literature does. ``against="target"`` scores a fixed class instead, which
    cancels itself out on any dataset with mixed predictions -- see
    :func:`class_scores`.
    """
    curve = deletion_curve(
        model_or_fn, X, W, baseline=baseline, target=target, steps=steps,
        against=against,
    )
    return np.nanmean(curve, axis=1)


def faithfulness_correlation(
    model_or_fn,
    X,
    W,
    *,
    baseline,
    target: Optional[int] = 1,
    subset_size: Optional[int] = None,
    n_subsets: int = 50,
    random_state: int = 0,
    against: str = "predicted",
) -> np.ndarray:
    """Faithfulness Correlation (Bhatt et al. 2020), per instance.

    Repeatedly mask a random feature subset and correlate the summed attribution
    of that subset with the drop it causes in the model's score. An explanation
    whose magnitudes track the model's behaviour scores near 1; one that ranks
    features arbitrarily scores near 0.

    Returns NaN for an instance whose drops or attribution sums are constant
    across subsets -- correlation is undefined there, and reporting 0 would read
    as "measured, and unfaithful".
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    base_row = np.asarray(baseline, dtype=float).reshape(-1)
    n_rows, n_features = X_arr.shape
    size = max(1, n_features // 2) if subset_size is None else int(subset_size)
    size = min(size, n_features)

    rng = np.random.default_rng(random_state)
    labels, target_scores, _space = model_scores(model_or_fn, X_arr, target=target)
    if against == "predicted":
        original = class_scores(model_or_fn, X_arr, labels)
        scored_class = labels
    else:
        original, scored_class = target_scores, None
    if original is None:
        return np.full(n_rows, np.nan)

    sums = np.empty((n_rows, n_subsets), dtype=float)
    drops = np.empty((n_rows, n_subsets), dtype=float)
    for s in range(n_subsets):
        columns = rng.choice(n_features, size=size, replace=False)
        masked = X_arr.copy()
        masked[:, columns] = base_row[columns]
        if scored_class is None:
            _l, scores, _sp = model_scores(model_or_fn, masked, target=target)
        else:
            scores = class_scores(model_or_fn, masked, scored_class)
        drops[:, s] = original - (scores if scores is not None else original)
        # Attributions explain the target class, so on an instance predicted the
        # other way the sign must be flipped to line up with its own drop.
        signed = W_arr[:, columns].sum(axis=1)
        sums[:, s] = signed if scored_class is None else np.where(labels == 1, signed, -signed)

    out = np.full(n_rows, np.nan)
    for i in range(n_rows):
        if np.std(sums[i]) == 0.0 or np.std(drops[i]) == 0.0:
            continue
        out[i] = float(np.corrcoef(sums[i], drops[i])[0, 1])
    return out


# -- complexity ------------------------------------------------------------


def sparsity_loss(W, *, atol: float = 1e-12) -> np.ndarray:
    """Count non-zero attribution weights per instance (effective complexity).

    Informative only for methods that genuinely zero features out -- a sparse
    logistic-regression surrogate, a rule set, or sim2real's ``sparse`` property,
    where it orders the four properties exactly as intended (sparse 1.0,
    sparse_robust 2.0, faithful 3.0, robust 4.0).

    For dense attribution methods it saturates: LIME and SHAP essentially never
    return an exact zero, so on adult's five raw features both score 5.0. That
    is a true statement about those methods, not a measurement failure -- but it
    means the count cannot separate them, and :func:`sparsity_gini` is the
    column to read there (0.52 vs 0.44 on the same run). Raising ``atol`` does
    not rescue it: thresholding at 1% of each instance's peak attribution still
    leaves LIME at 7.94 and SHAP at 7.91 of 9 model-space columns.
    """
    weights = ensure_2d(W)
    return np.sum(np.abs(weights) > atol, axis=1).astype(float)


def sparsity_gini(W) -> np.ndarray:
    """Gini coefficient of |attributions| per instance (Chalasani et al. 2020).

    0 when every feature carries equal weight, approaching 1 when one feature
    carries everything. Unlike a non-zero count this is comparable across
    datasets with different feature counts, and it separates methods like SHAP
    that rarely produce an exact zero.
    """
    weights = np.abs(ensure_2d(W))
    n_features = weights.shape[1]
    out = np.zeros(weights.shape[0], dtype=float)
    for i, row in enumerate(weights):
        total = row.sum()
        if total <= 0.0 or n_features < 2:
            continue
        ordered = np.sort(row)
        index = np.arange(1, n_features + 1)
        out[i] = float(
            (2.0 * np.sum(index * ordered)) / (n_features * total) - (n_features + 1.0) / n_features
        )
    return out


def complexity_entropy(W) -> np.ndarray:
    """Shannon entropy of the normalised |attribution| distribution (Bhatt 2020).

    Lower is simpler: a one-feature explanation scores 0, a uniform one scores
    ``log(n_features)``.
    """
    weights = np.abs(ensure_2d(W))
    out = np.zeros(weights.shape[0], dtype=float)
    for i, row in enumerate(weights):
        total = row.sum()
        if total <= 0.0:
            continue
        fractions = row[row > 0.0] / total
        out[i] = float(-np.sum(fractions * np.log(fractions)))
    return out


# -- robustness ------------------------------------------------------------


def _resolve_radius(
    distances: np.ndarray, radius: Optional[float], radius_quantile: float
) -> float:
    if radius is not None:
        return float(radius)
    positive = distances[distances > 0.0]
    if positive.size == 0:
        return 0.0
    resolved = float(np.quantile(positive, radius_quantile))
    return resolved if resolved > 0.0 else float(positive.min())


def robustness_loss(
    X,
    W,
    *,
    radius: Optional[float] = None,
    radius_quantile: float = 0.10,
    neighbor_pool_size: int = 1000,
    random_state: int = 0,
    empty_neighborhood: str = "zero",
) -> np.ndarray:
    """Local Lipschitz estimate (Alvarez-Melis & Jaakkola 2018), per instance.

    ``max ||w_j - w_i|| / ||x_j - x_i||`` over neighbours within ``radius``.
    Lower means the explanation moves less than the input does.

    ``radius=None`` resolves the radius from the data -- the
    ``radius_quantile`` quantile of the observed positive pairwise distances --
    because a fixed radius in raw feature units is meaningless across datasets,
    and the historical default of 1.0 spans the whole range of min-max scaled
    data, making every instance a "neighbour".

    ``empty_neighborhood="zero"`` reports 0.0 for an instance with no neighbour,
    which is what the sim2real search expects but reads as "perfectly stable";
    ``"nan"`` marks it unmeasured instead.

    The neighbour pool is capped at ``neighbor_pool_size`` (seeded) so the
    ``O(n^2)`` comparison stays affordable on a 22k-row dataset. Every instance
    is still scored -- only the pool it is compared against is subsampled.
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    n_rows = X_arr.shape[0]

    if n_rows > neighbor_pool_size:
        rng = np.random.default_rng(random_state)
        pool = rng.choice(n_rows, size=neighbor_pool_size, replace=False)
    else:
        pool = np.arange(n_rows)
    X_pool, W_pool = X_arr[pool], W_arr[pool]

    if radius is None:
        sample = X_pool[: min(len(pool), 200)]
        pairwise = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
        resolved_radius = _resolve_radius(pairwise, None, radius_quantile)
    else:
        resolved_radius = float(radius)

    empty = 0.0 if empty_neighborhood == "zero" else np.nan
    losses = np.full(n_rows, empty, dtype=float)
    for i in range(n_rows):
        distances = np.linalg.norm(X_pool - X_arr[i], axis=1)
        mask = (distances > 0.0) & (distances <= resolved_radius)
        if not np.any(mask):
            continue
        weight_distances = np.linalg.norm(W_pool[mask] - W_arr[i], axis=1)
        losses[i] = float(np.max(weight_distances / distances[mask]))
    return losses



def relative_input_stability(
    X,
    W,
    *,
    radius: Optional[float] = None,
    radius_quantile: float = 0.10,
    neighbor_pool_size: int = 1000,
    random_state: int = 0,
    eps_min: float = 1e-6,
) -> np.ndarray:
    """Relative Input Stability (Agarwal et al. 2022), per instance.

    ``max_j || (w_i - w_j) / w_i || / max(|| (x_i - x_j) / x_i ||, eps_min)``

    The local Lipschitz estimate divides an *absolute* explanation difference by
    an absolute input difference, so its value carries the units of whatever the
    explanation is measured in. That makes it a fair comparison between two runs
    of the same method and an unfair one between methods: on adult, LIME scores
    3.87 and SHAP 0.43, which mixes a genuine stability difference with the fact
    that LIME's local-surrogate coefficients and SHAP's probability-space
    contributions are not the same quantity.

    RIS was introduced to fix exactly that: both differences are made *relative*
    to their own magnitudes before dividing, so the units cancel and the result
    is comparable across methods. Quantus reports it alongside the Lipschitz
    estimate for the same reason, so both are kept here.

    Lower is more stable. ``eps_min`` guards the division where a feature or an
    attribution is zero.
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    n_rows = X_arr.shape[0]

    if n_rows > neighbor_pool_size:
        rng = np.random.default_rng(random_state)
        pool = rng.choice(n_rows, size=neighbor_pool_size, replace=False)
    else:
        pool = np.arange(n_rows)
    X_pool, W_pool = X_arr[pool], W_arr[pool]

    if radius is None:
        sample = X_pool[: min(len(pool), 200)]
        pairwise = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
        radius = _resolve_radius(pairwise, None, radius_quantile)

    out = np.full(n_rows, np.nan, dtype=float)
    for i in range(n_rows):
        distances = np.linalg.norm(X_pool - X_arr[i], axis=1)
        mask = (distances > 0.0) & (distances <= radius)
        if not np.any(mask):
            continue
        x_denominator = np.where(np.abs(X_arr[i]) < eps_min, eps_min, X_arr[i])
        w_denominator = np.where(np.abs(W_arr[i]) < eps_min, eps_min, W_arr[i])
        input_change = np.linalg.norm((X_arr[i] - X_pool[mask]) / x_denominator, axis=1)
        explanation_change = np.linalg.norm((W_arr[i] - W_pool[mask]) / w_denominator, axis=1)
        out[i] = float(np.max(explanation_change / np.maximum(input_change, eps_min)))
    return out


def robustness_loss_with_radius(X, W, **kwargs) -> tuple[np.ndarray, float]:
    """``robustness_loss`` plus the radius it actually used."""
    X_arr = ensure_2d(X)
    radius = kwargs.get("radius")
    if radius is None:
        sample = X_arr[: min(X_arr.shape[0], 200)]
        pairwise = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
        radius = _resolve_radius(pairwise, None, kwargs.get("radius_quantile", 0.10))
    return robustness_loss(X, W, **{**kwargs, "radius": radius}), float(radius)


# -- the whole set ---------------------------------------------------------


def summarize_property_metrics(
    model_or_fn,
    X,
    W,
    *,
    output_type: str,
    base_values: Optional[np.ndarray] = None,
    radius: float = 1.0,
) -> dict[str, float]:
    """Return mean sparsity, faithfulness loss, and robustness loss.

    Signature and defaults unchanged: ``Sim2RealPropertyAttribution`` calls this
    with coefficient-convention weights against a synthetic function.
    """
    if output_type == "regression":
        faithfulness = fidelity_regression_loss(model_or_fn, X, W, base_values=base_values)
    else:
        faithfulness = fidelity_classification_loss(model_or_fn, X, W, base_values=base_values)
    return {
        "faithfulness_loss": float(np.mean(faithfulness)),
        "sparsity_loss": float(np.mean(sparsity_loss(W))),
        "robustness_loss": float(np.mean(robustness_loss(X, W, radius=radius))),
    }


def explanation_quality(
    model_or_fn,
    X,
    W,
    *,
    spec: QualitySpec,
    base_values: Optional[np.ndarray] = None,
    baseline=None,
    target: Optional[int] = 1,
    shown_values=None,
    robustness_radius: Optional[float] = None,
    robustness_radius_quantile: float = 0.10,
    neighbor_pool_size: int = 1000,
    n_subsets: int = 50,
    random_state: int = 0,
) -> dict[str, Any]:
    """Every quality metric for one method's explanations, per instance.

    ``X``/``W`` are model-space (pre-aggregation) so faithfulness and robustness
    are measured against the vector the model actually saw. ``shown_values`` is
    the post-aggregation vector a participant reads, which is what complexity
    should describe; it defaults to ``W``.
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    shown = W_arr if shown_values is None else ensure_2d(shown_values)
    notes: list[str] = []

    if baseline is None:
        baseline = np.mean(X_arr, axis=0)

    _labels, _scores, space = model_scores(model_or_fn, X_arr, target=target)
    if space == "multiclass":
        n = X_arr.shape[0]
        aopc = np.full(n, np.nan)
        corr = np.full(n, np.nan)
        notes.append("multiclass output: no single positive class to score against")
    else:
        aopc = faithfulness_aopc(model_or_fn, X_arr, W_arr, baseline=baseline, target=target)
        corr = faithfulness_correlation(
            model_or_fn, X_arr, W_arr, baseline=baseline, target=target,
            n_subsets=n_subsets, random_state=random_state,
        )

    if spec.fidelity == "none":
        recon = np.full(X_arr.shape[0], np.nan)
        if spec.note:
            notes.append(spec.note)
    elif spec.fidelity == "label":
        recon = fidelity_classification_loss(
            model_or_fn, X_arr, W_arr, base_values=base_values,
            convention=spec.convention, link=spec.link, target=target,
        )
    else:
        recon = fidelity_regression_loss(
            model_or_fn, X_arr, W_arr, base_values=base_values,
            convention=spec.convention, link=spec.link, target=target,
        )

    lipschitz, resolved_radius = robustness_loss_with_radius(
        X_arr, W_arr,
        radius=robustness_radius,
        radius_quantile=robustness_radius_quantile,
        neighbor_pool_size=neighbor_pool_size,
        random_state=random_state,
        empty_neighborhood="nan",
    )

    ris = relative_input_stability(
        X_arr, W_arr,
        radius=robustness_radius,
        radius_quantile=robustness_radius_quantile,
        neighbor_pool_size=neighbor_pool_size,
        random_state=random_state,
    )

    return {
        "faithfulness_aopc": aopc,
        "faithfulness_corr": corr,
        "faithfulness_loss": recon,
        "sparsity_nonzero": sparsity_loss(shown),
        "sparsity_gini": sparsity_gini(shown),
        "complexity_entropy": complexity_entropy(shown),
        "robustness_lipschitz": lipschitz,
        "robustness_ris": ris,
        "quality_note": "; ".join(notes),
        "score_space": space,
        "robustness_radius": resolved_radius,
        "convention": spec.convention if spec.fidelity != "none" else "indicator",
    }


def score_result(
    result,
    model_or_fn,
    X,
    *,
    shown_values=None,
    target: Optional[int] = 1,
    spec: Optional[QualitySpec] = None,
    overrides: Optional[Mapping[str, QualitySpec]] = None,
    **kwargs,
) -> dict[str, Any]:
    """Score any adapter's ``XAIAdapterResult``, whichever method produced it.

    This is the entry point for adapters outside this package. It reads the
    method name, values and base values off the result, so:

        from src.xai_adapter import score_result
        result = my_adapter.explain(X)
        quality = score_result(result, model, X)

    No registration is required. Faithfulness (AOPC and correlation), complexity
    and robustness need only the attribution *ranking*, so they are computed for
    a method this module has never heard of. Declaring a
    :class:`QualitySpec` -- via ``spec=``, ``overrides=`` or
    :func:`register_quality_spec` -- additionally unlocks the exact
    reconstruction loss, which is the one metric that must know whether the
    values are coefficients or contributions.
    """
    resolved = spec or quality_spec(getattr(result, "method", ""), overrides=overrides)
    return explanation_quality(
        model_or_fn,
        X,
        getattr(result, "values", result),
        spec=resolved,
        base_values=getattr(result, "base_values", None),
        target=target,
        shown_values=shown_values,
        **kwargs,
    )


def quality_table(explanations, *, by: str = "expMethod", scores: bool = False):
    """Mean quality per XAI method, from a table the explanation stage produced.

    Rows without quality columns (prediction-only rows, or a run with the
    metrics switched off) are excluded rather than counted as zeros. With
    ``scores=True`` each metric also gets a ``*_score`` column, min-max
    normalised across methods and oriented so higher is better -- a display
    convenience, not a publishable quantity, since the normalisation depends on
    which methods happen to be in the table.
    """
    import pandas as pd

    frame = pd.DataFrame(explanations)
    present = [column for column in QUALITY_METRIC_COLUMNS if column in frame.columns]
    if not present or by not in frame.columns:
        return pd.DataFrame()

    scored = frame.dropna(subset=present, how="all")
    if scored.empty:
        return pd.DataFrame()

    table = scored.groupby(by, dropna=False)[present].mean()
    table.insert(0, "n_rows", scored.groupby(by, dropna=False).size())
    if "quality_note" in scored.columns:
        notes = scored.groupby(by, dropna=False)["quality_note"].agg(
            lambda values: next((v for v in values if isinstance(v, str) and v), "")
        )
        table["quality_note"] = notes

    if scores:
        for column in present:
            values = table[column].astype(float)
            spread = values.max() - values.min()
            if not np.isfinite(spread) or spread == 0.0:
                table[f"{column}_score"] = np.where(values.notna(), 1.0, np.nan)
                continue
            normalised = (values - values.min()) / spread
            table[f"{column}_score"] = (
                normalised if column in _HIGHER_IS_BETTER else 1.0 - normalised
            )
    return table.reset_index()

"""Technical quality metrics for explanations.

``src/xai_adapter/metrics.py`` shipped with zero tests and three convention bugs
that could only ever surface as plausible-looking numbers, never as errors:

1. both fidelity functions reconstructed ``base + sum(x * w)`` -- a *coefficient*
   convention -- while SHAP/LIME produce *contributions*, where the correct
   reconstruction is ``base + sum(w)``;
2. ``_predict`` flattened the model's output with ``reshape(-1)``, but this
   repo's models return ``(n, 2)`` scores, yielding ``2n`` values compared
   against ``n`` reconstructions;
3. the classification threshold was ``> 0`` (margin space), so on probability
   outputs every instance was called class 1.

The module's one existing consumer, ``Sim2RealPropertyAttribution``, depends on
the old defaults -- it random-searches attribution matrices against synthetic
models that return labels and true linear coefficients -- so those defaults are
pinned here too.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xai_adapter.metrics import (
    QualitySpec,
    complexity_entropy,
    deletion_curve,
    explanation_quality,
    faithfulness_aopc,
    faithfulness_correlation,
    fidelity_classification_loss,
    fidelity_regression_loss,
    model_scores,
    quality_spec,
    quality_table,
    robustness_loss,
    robustness_loss_with_radius,
    sparsity_gini,
    sparsity_loss,
)


class _AdditiveProbabilityModel:
    """Score is exactly ``0.5 + 0.1*x0 + 0.2*x1``, returned as a 1-D probability."""

    coefficients = np.array([0.1, 0.2])
    base = 0.5

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(self.base + X @ self.coefficients, 0.0, 1.0)


class _StraddlingProbabilityModel:
    """Score ``0.2 + 0.6*x0`` -- crosses 0.5, so both classes actually occur.

    Needed to tell the thresholds apart: a model whose probability never dips
    below 0.5 labels everything class 1, and then a ``> 0`` test agrees with a
    ``> 0.5`` test by accident.
    """

    coefficient = 0.6
    base = 0.2

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(self.base + self.coefficient * X[:, 0], 0.0, 1.0)


class _TwoColumnModel:
    """Returns ``(n, 2)`` softmax rows, like this repo's torch/Keras MLPs."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        margin = X[:, 0] - 0.5
        positive = 1.0 / (1.0 + np.exp(-4.0 * margin))
        return np.column_stack([1.0 - positive, positive])


class _LabelModel:
    """Returns integer labels, like the sim2real synthetic functions."""

    coefficients = np.array([1.0, -1.0])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.coefficients - 0.5 > 0.0).astype(int)


class _FirstFeatureOnlyModel:
    """Depends on x0 alone, so the faithful ranking is unambiguous."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(0.5 + 0.4 * X[:, 0], 0.0, 1.0)


class _MulticlassModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.tile(np.array([0.2, 0.3, 0.5]), (X.shape[0], 1))


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(24, 2))


# -- reading the model's output (bug 2) -----------------------------------


def test_two_column_model_output_is_reduced_to_the_target_class_not_flattened(X):
    """reshape(-1) on (n, 2) gave 2n values compared against n reconstructions."""
    labels, scores, space = model_scores(_TwoColumnModel(), X, target=1)

    assert labels.shape == (len(X),)
    assert scores is not None and scores.shape == (len(X),)
    assert space == "probability"
    # The selected column really is the positive class, not column 0.
    raw = _TwoColumnModel().predict(X)
    assert np.allclose(scores, raw[:, 1])


def test_a_label_only_model_reports_no_scores(X):
    labels, scores, space = model_scores(_LabelModel(), X)
    assert space == "label"
    assert scores is None
    assert set(labels.tolist()) <= {0, 1}


# -- faithfulness: reconstruction ------------------------------------------


def test_a_perfectly_additive_contribution_explanation_scores_zero_loss(X):
    model = _AdditiveProbabilityModel()
    contributions = X * model.coefficients  # phi_i, already the effect
    base = np.full(len(X), model.base)

    loss = fidelity_regression_loss(
        model, X, contributions, base_values=base, convention="contribution"
    )
    assert loss == pytest.approx(np.zeros(len(X)), abs=1e-12)


def test_reading_contributions_as_coefficients_calls_a_perfect_explanation_wrong(X):
    """Bug 1, pinned: the same explanation scored under the wrong convention."""
    model = _AdditiveProbabilityModel()
    contributions = X * model.coefficients
    base = np.full(len(X), model.base)

    wrong = fidelity_regression_loss(
        model, X, contributions, base_values=base, convention="coefficient"
    )
    assert wrong.mean() > 1e-6


def test_sim2real_ground_truth_coefficients_still_score_zero_by_default(X):
    """The existing consumer must keep working: coefficients, labels, threshold 0."""
    model = _LabelModel()
    weights = np.tile(model.coefficients, (len(X), 1))
    base = np.full(len(X), -0.5)

    # No convention keyword -- exactly how sim2real.py calls it.
    loss = fidelity_classification_loss(model, X, weights, base_values=base)
    assert loss == pytest.approx(np.zeros(len(X)))


def test_probability_outputs_are_thresholded_at_a_half_not_at_zero(X):
    """Bug 3: with a > 0 test every probability reconstruction is class 1."""
    model = _StraddlingProbabilityModel()
    contributions = np.column_stack([model.coefficient * X[:, 0], np.zeros(len(X))])
    base = np.full(len(X), model.base)
    # Both classes really are present, or the two thresholds cannot differ.
    assert set(np.asarray(model.predict(X) >= 0.5, dtype=int).tolist()) == {0, 1}

    correct = fidelity_classification_loss(
        model, X, contributions, base_values=base, convention="contribution"
    )
    forced_margin = fidelity_classification_loss(
        model, X, contributions, base_values=base,
        convention="contribution", threshold=0.0,
    )
    # Every reconstruction exceeds 0, so the margin threshold calls everything
    # class 1 and disagrees wherever the model says class 0.
    assert forced_margin.sum() > correct.sum()


# -- faithfulness: perturbation-based (the standard) -----------------------


def test_deleting_the_top_attributed_feature_moves_the_score_most(X):
    model = _FirstFeatureOnlyModel()
    faithful = np.tile([1.0, 0.0], (len(X), 1))
    curve = deletion_curve(model, X, faithful, baseline=np.mean(X, axis=0))
    # First step masks x0 (the only feature that matters), so the drop is real.
    assert np.abs(curve[:, 0]).mean() > 0.0


def test_a_faithful_ranking_scores_higher_aopc_than_a_reversed_one(X):
    model = _FirstFeatureOnlyModel()
    baseline = np.mean(X, axis=0)
    faithful = np.tile([1.0, 0.0], (len(X), 1))
    reversed_ranking = np.tile([0.0, 1.0], (len(X), 1))

    good = faithfulness_aopc(model, X, faithful, baseline=baseline)
    bad = faithfulness_aopc(model, X, reversed_ranking, baseline=baseline)
    assert np.nanmean(np.abs(good)) > np.nanmean(np.abs(bad))


def test_faithfulness_is_measurable_without_knowing_the_convention(X):
    """The point of the perturbation metrics: ranking is all they need."""
    model = _FirstFeatureOnlyModel()
    baseline = np.mean(X, axis=0)
    contributions = np.tile([1.0, 0.0], (len(X), 1))
    scaled = contributions * 1000.0  # same ranking, wildly different units

    a = faithfulness_aopc(model, X, contributions, baseline=baseline)
    b = faithfulness_aopc(model, X, scaled, baseline=baseline)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_an_uninformative_explanation_scores_lower_correlation_than_a_faithful_one(X):
    model = _AdditiveProbabilityModel()
    baseline = np.mean(X, axis=0)
    faithful = X * model.coefficients
    rng = np.random.default_rng(3)
    noise = rng.normal(size=faithful.shape)

    good = faithfulness_correlation(model, X, faithful, baseline=baseline, random_state=0)
    bad = faithfulness_correlation(model, X, noise, baseline=baseline, random_state=0)
    assert np.nanmean(good) > np.nanmean(bad)


# -- complexity -------------------------------------------------------------


def test_sparsity_counts_the_non_zero_weights():
    weights = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert sparsity_loss(weights) == pytest.approx([1.0, 3.0])


def test_a_one_feature_explanation_is_maximally_sparse_and_minimally_complex():
    concentrated = np.array([[5.0, 0.0, 0.0, 0.0]])
    uniform = np.array([[1.0, 1.0, 1.0, 1.0]])

    assert sparsity_gini(concentrated)[0] > sparsity_gini(uniform)[0]
    assert complexity_entropy(concentrated)[0] == pytest.approx(0.0)
    assert complexity_entropy(uniform)[0] == pytest.approx(np.log(4.0))


def test_gini_is_comparable_where_a_non_zero_count_is_not():
    """SHAP rarely produces an exact zero, so the count saturates; Gini does not."""
    almost_sparse = np.array([[10.0, 1e-6, 1e-6, 1e-6]])
    assert sparsity_loss(almost_sparse)[0] == 4.0        # count says "dense"
    assert sparsity_gini(almost_sparse)[0] > 0.7          # Gini says "concentrated"


# -- robustness -------------------------------------------------------------


def test_the_radius_defaults_to_a_quantile_of_the_observed_distances(X):
    _losses, radius = robustness_loss_with_radius(X, np.ones_like(X))
    pairwise = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    positive = pairwise[pairwise > 0.0]
    assert radius == pytest.approx(float(np.quantile(positive, 0.10)), rel=1e-6)
    # The historical default of 1.0 spans this whole space, making it non-local.
    assert radius < 1.0


def test_an_instance_with_no_neighbour_is_not_reported_as_perfectly_stable():
    far_apart = np.array([[0.0, 0.0], [100.0, 100.0]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])

    marked = robustness_loss(far_apart, weights, radius=0.001, empty_neighborhood="nan")
    assert np.all(np.isnan(marked))
    # sim2real's default still reports 0.0, which its search expects.
    legacy = robustness_loss(far_apart, weights, radius=0.001)
    assert legacy == pytest.approx([0.0, 0.0])


def test_the_neighbour_pool_is_capped_and_deterministic_under_a_seed():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(200, 3))
    W = rng.uniform(size=(200, 3))

    first = robustness_loss(X, W, radius=0.5, neighbor_pool_size=20, random_state=7)
    again = robustness_loss(X, W, radius=0.5, neighbor_pool_size=20, random_state=7)

    assert first.shape == (200,)          # every instance still scored
    assert np.allclose(first, again, equal_nan=True)


# -- specs and the whole set ------------------------------------------------


def test_known_methods_declare_how_their_numbers_must_be_read():
    assert quality_spec("shap").convention == "contribution"
    assert quality_spec("logistic_regression").link == "logit"
    assert quality_spec("sim2real_property").convention == "coefficient"
    # Structure, not magnitude: no reconstruction is meaningful.
    assert quality_spec("decision_tree").fidelity == "none"


def test_an_unknown_method_is_not_scored_on_a_guessed_convention():
    spec = quality_spec("something_new")
    assert spec.fidelity == "none"
    assert "not declared" in spec.note


def test_a_multiclass_model_leaves_faithfulness_unscored_rather_than_guessing(X):
    quality = explanation_quality(
        _MulticlassModel(), X, np.ones_like(X), spec=QualitySpec(fidelity="none")
    )
    assert np.all(np.isnan(quality["faithfulness_aopc"]))
    assert "multiclass" in quality["quality_note"]


def test_explanation_quality_returns_one_value_per_instance(X):
    model = _AdditiveProbabilityModel()
    quality = explanation_quality(
        model, X, X * model.coefficients,
        spec=quality_spec("shap"),
        base_values=np.full(len(X), model.base),
    )
    for column in (
        "faithfulness_aopc", "faithfulness_corr", "faithfulness_loss",
        "sparsity_nonzero", "sparsity_gini", "complexity_entropy",
        "robustness_lipschitz",
    ):
        assert np.asarray(quality[column]).shape == (len(X),), column
    assert quality["faithfulness_loss"] == pytest.approx(np.zeros(len(X)), abs=1e-12)


def test_complexity_describes_the_vector_the_participant_sees(X):
    """Faithfulness uses model-space values; complexity uses the shown ones."""
    model = _AdditiveProbabilityModel()
    shown = np.tile([1.0, 0.0], (len(X), 1))  # one visible feature after aggregation

    quality = explanation_quality(
        model, X, X * model.coefficients, spec=quality_spec("shap"), shown_values=shown
    )
    assert quality["sparsity_nonzero"] == pytest.approx(np.ones(len(X)))


def test_the_summary_table_averages_each_method_separately():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "expMethod": ["lime", "lime", "shap", "shap"],
            "faithfulness_aopc": [0.1, 0.3, 0.5, 0.7],
            "faithfulness_corr": [0.2, 0.2, 0.8, 0.8],
            "faithfulness_loss": [1.0, 1.0, 0.0, 0.0],
            "sparsity_nonzero": [2.0, 2.0, 4.0, 4.0],
            "sparsity_gini": [0.5, 0.5, 0.1, 0.1],
            "complexity_entropy": [0.3, 0.3, 1.2, 1.2],
            "robustness_lipschitz": [1.0, 1.0, 2.0, 2.0],
        }
    )
    table = quality_table(frame)
    lime = table[table["expMethod"] == "lime"].iloc[0]
    assert lime["faithfulness_aopc"] == pytest.approx(0.2)
    assert lime["n_rows"] == 2

    scored = quality_table(frame, scores=True)
    # shap has the higher AOPC, so it must score 1.0 on the higher-is-better view.
    shap_row = scored[scored["expMethod"] == "shap"].iloc[0]
    assert shap_row["faithfulness_aopc_score"] == pytest.approx(1.0)
    # ...and the lower reconstruction loss must also score 1.0 once inverted.
    assert shap_row["faithfulness_loss_score"] == pytest.approx(1.0)


def test_rows_without_quality_columns_are_excluded_not_counted_as_zero():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "expMethod": ["__prediction_only__", "lime"],
            "faithfulness_aopc": [np.nan, 0.4],
            "sparsity_nonzero": [np.nan, 2.0],
        }
    )
    table = quality_table(frame)
    assert list(table["expMethod"]) == ["lime"]


# -- usable by any adapter, not just the ones declared here ----------------


def test_any_adapters_result_can_be_scored_without_registration(X):
    """A method this module has never heard of still gets the standard metrics.

    Faithfulness (AOPC/correlation), complexity and robustness need only the
    attribution *ranking*, never its units -- which is exactly why the
    perturbation family was chosen over the reconstruction loss.
    """
    from src.xai_adapter import score_result
    from src.xai_adapter.base import XAIAdapterResult

    result = XAIAdapterResult(
        values=np.tile([1.0, 0.0], (len(X), 1)),
        base_values=None,
        method="a_method_nobody_declared",
        metadata={},
    )
    quality = score_result(result, _FirstFeatureOnlyModel(), X)

    for column in ("faithfulness_aopc", "sparsity_nonzero", "sparsity_gini",
                   "complexity_entropy", "robustness_lipschitz"):
        assert np.isfinite(np.asarray(quality[column])).any(), column
    # Only the convention-dependent reconstruction declines to guess.
    assert np.all(np.isnan(quality["faithfulness_loss"]))
    assert "not declared" in quality["quality_note"]


def test_registering_a_spec_unlocks_the_reconstruction_loss(X):
    from src.xai_adapter import QualitySpec, register_quality_spec, score_result
    from src.xai_adapter.base import XAIAdapterResult

    model = _AdditiveProbabilityModel()
    result = XAIAdapterResult(
        values=X * model.coefficients,
        base_values=np.full(len(X), model.base),
        method="my_custom_attributor",
        metadata={},
    )
    register_quality_spec(
        "my_custom_attributor", QualitySpec("contribution", "identity", "score")
    )

    quality = score_result(result, model, X)
    assert quality["faithfulness_loss"] == pytest.approx(np.zeros(len(X)), abs=1e-12)


def test_a_spec_can_be_passed_per_call_without_touching_the_registry(X):
    from src.xai_adapter import QualitySpec, score_result
    from src.xai_adapter.base import XAIAdapterResult

    model = _AdditiveProbabilityModel()
    result = XAIAdapterResult(
        values=X * model.coefficients,
        base_values=np.full(len(X), model.base),
        method="one_off_method",
        metadata={},
    )
    quality = score_result(
        result, model, X, spec=QualitySpec("contribution", "identity", "score")
    )
    assert quality["faithfulness_loss"] == pytest.approx(np.zeros(len(X)), abs=1e-12)


def test_a_variant_suffixed_method_name_resolves_to_its_base_spec():
    """Adapters name results like 'sim2real_property_faithful'."""
    from src.xai_adapter import quality_spec

    assert quality_spec("sim2real_property_faithful").convention == "coefficient"
    assert quality_spec("sim2real_property_sparse_robust").convention == "coefficient"


# -- AOPC must not cancel itself out on mixed predictions ------------------


class _MixedPredictionModel:
    """Predicts class 0 for most instances, class 1 for a few -- like adult."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        positive = 1.0 / (1.0 + np.exp(-8.0 * (X[:, 0] - 0.75)))
        return np.column_stack([1.0 - positive, positive])


def test_aopc_measures_the_predicted_class_not_a_fixed_one():
    """Regression: scoring a fixed class averages a faithful explanation to zero.

    Masking the decisive feature lowers p(class 1) for instances predicted 1 and
    *raises* it for instances predicted 0. On adult -- 230 of 300 instances
    predicted 0 -- the two halves cancelled: +0.43 and -0.14 averaged to -0.008,
    while both behaved exactly as a faithful explanation should. SHAP's AOPC came
    out below LIME's, contradicting its much higher faithfulness correlation,
    which is what exposed the bug.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    model = _MixedPredictionModel()
    faithful = np.tile([1.0, 0.0], (len(X), 1))
    baseline = np.mean(X, axis=0)

    labels, _scores, _space = model_scores(model, X)
    assert 0 in set(labels.tolist()) and 1 in set(labels.tolist()), "need mixed predictions"

    predicted = faithfulness_aopc(model, X, faithful, baseline=baseline)
    fixed = faithfulness_aopc(model, X, faithful, baseline=baseline, against="target")

    # Scoring the predicted class, a faithful explanation is positive nearly
    # everywhere; scoring a fixed class, the two halves fight each other.
    assert np.nanmean(predicted) > 0.0
    # A clear majority, not a specific fraction -- the exact share depends on how
    # sharp the fixture's decision boundary is. On real adult data it is 83%.
    assert (predicted > 0).mean() > 0.5
    assert np.nanmean(predicted) > np.nanmean(fixed)


def test_the_two_halves_of_a_fixed_class_score_have_opposite_signs():
    """The mechanism itself, so a future change cannot quietly reintroduce it."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    model = _MixedPredictionModel()
    faithful = np.tile([1.0, 0.0], (len(X), 1))

    labels, _s, _sp = model_scores(model, X)
    fixed = faithfulness_aopc(
        model, X, faithful, baseline=np.mean(X, axis=0), against="target"
    )
    assert np.nanmean(fixed[labels == 1]) > 0.0
    assert np.nanmean(fixed[labels == 0]) < 0.0

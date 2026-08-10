import math
from pathlib import Path

import numpy as np
import pytest

from src.cognitive_models.cognitive_models.sim2real.gcm_strategies import (
    SIM2REAL_RAW_FEATURE_ORDER,
    Sim2RealAttributionProjector,
    Sim2RealFittedAttributionSum,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def projector():
    return Sim2RealAttributionProjector.from_csv_files(
        values_csv=ROOT / "assets/ai_dataset/sim2real/values.csv",
        attribution_csv=(
            ROOT / "assets/explanations/xai_desiderata/attribution.csv"
        ),
        counterfactual_csv=(
            ROOT / "assets/explanations/xai_desiderata/counterfactuals_fake.csv"
        ),
    )


def test_projects_every_variant_to_twelve_raw_features(projector):
    for exp_property in (None, "faithful", "robust", "sparse", "sparse_robust"):
        pairs = projector.project_all(exp_property=exp_property)
        assert len(pairs) == 39
        assert all(pair.feature_names == SIM2REAL_RAW_FEATURE_ORDER for pair in pairs)
        assert all(pair.original_attributions.shape == (12,) for pair in pairs)
        assert all(pair.changed_attributions.shape == (12,) for pair in pairs)


def test_categorical_change_selects_new_one_hot_attribution(projector):
    pair = projector.project(0, exp_property="faithful")
    relationship = pair.feature_names.index("relationship")

    assert pair.original_dimension_indices[relationship] == 51
    assert pair.changed_dimension_indices[relationship] == 55
    assert pair.original_attributions[relationship] == pytest.approx(0.02)
    assert pair.changed_attributions[relationship] == pytest.approx(0.03)
    assert pair.changed_feature_names == ("relationship",)


def test_corrected_instance_one_changes_female_to_male(projector):
    pair = projector.project(1, exp_property="faithful")
    sex = pair.feature_names.index("sex")

    assert pair.original_dimension_indices[sex] == 56
    assert pair.changed_dimension_indices[sex] == 57
    assert pair.changed_feature_names == ("sex",)
    assert pair.qid == 14
    assert pair.ai_prediction == 0
    assert pair.correct_increase_label == 1
    assert pair.split == "training"


def test_confidence_delta_uses_new_minus_original(projector):
    pair = projector.project(0, exp_property="faithful")
    result = Sim2RealFittedAttributionSum().compare(pair)

    assert result.confidence_delta == pytest.approx(
        result.p_new_high_income - result.p_original_high_income
    )
    assert result.confidence_delta > 0.0
    assert result.probability_income_increases > 0.5
    assert result.qid == 12


def test_lapse_mixes_strategy_probability_with_unbiased_guessing(projector):
    pair = projector.project(0, exp_property="sparse")
    no_lapse = Sim2RealFittedAttributionSum().compare(pair)
    lapsed = Sim2RealFittedAttributionSum(lapse_rate=0.2).compare(pair)

    assert not lapsed.changed_feature_visible
    assert lapsed.effective_lapse_rate == pytest.approx(0.2)
    assert lapsed.guess_probability_income_increases == pytest.approx(0.5)
    assert lapsed.pre_lapse_probability_income_increases == pytest.approx(
        no_lapse.probability_income_increases
    )
    assert lapsed.probability_income_increases == pytest.approx(
        0.2 / 2.0 + 0.8 * no_lapse.probability_income_increases
    )


def test_guess_bias_controls_lapse_response_distribution(projector):
    pair = projector.project(0, exp_property="sparse")
    result = Sim2RealFittedAttributionSum(
        lapse_rate=1.0,
        guess_bias=np.log(3.0),
    ).compare(pair)

    assert result.guess_probability_income_increases == pytest.approx(0.75)
    assert result.probability_income_increases == pytest.approx(0.75)


def test_visible_changed_weight_bypasses_lapse(projector):
    pair = projector.project(0, exp_property="faithful")
    baseline = Sim2RealFittedAttributionSum().compare(pair)
    result = Sim2RealFittedAttributionSum(
        lapse_rate=1.0,
        guess_bias=np.log(3.0),
    ).compare(pair)

    assert result.changed_feature_visible
    assert result.effective_lapse_rate == pytest.approx(0.0)
    assert result.probability_income_increases == pytest.approx(
        baseline.probability_income_increases
    )


def test_lapse_and_guess_bias_are_validated():
    for lapse_rate in (-0.01, 1.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="lapse_rate"):
            Sim2RealFittedAttributionSum(lapse_rate=lapse_rate)
    for guess_bias in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="guess_bias"):
            Sim2RealFittedAttributionSum(guess_bias=guess_bias)


def test_numeric_change_can_use_value_weighted_mode(projector):
    pair = projector.project(2, exp_property="faithful")
    exact = Sim2RealFittedAttributionSum(aggregation="attribution").compare(pair)
    weighted = Sim2RealFittedAttributionSum(
        aggregation="value_weighted"
    ).compare(pair)

    assert pair.changed_feature_names == ("capital-gain",)
    assert exact.confidence_delta == pytest.approx(0.0)
    assert not np.isclose(weighted.confidence_delta, 0.0)


def test_max_features_attended_keeps_change_and_strongest_context(projector):
    pair = projector.project(0, exp_property="faithful")
    result = Sim2RealFittedAttributionSum(max_features_attended=2).compare(pair)

    assert "relationship" in result.attended_feature_names
    assert len(result.attended_feature_names) == 2


def test_max_features_attended_validates_range():
    for value in (0, 13, 1.5, True):
        with pytest.raises(ValueError, match="max_features_attended"):
            Sim2RealFittedAttributionSum(max_features_attended=value)


def test_truncated_source_attributions_require_explicit_policy(projector):
    pair = projector.project(9, exp_property="sparse_robust")

    with pytest.raises(ValueError, match="missing attributions"):
        Sim2RealFittedAttributionSum().compare(pair)

    result = Sim2RealFittedAttributionSum(
        missing_attributions="zero"
    ).compare(pair)
    assert np.isfinite(result.confidence_delta)


def test_projector_loads_higher_lower_labels_from_deltas(projector):
    labels = projector.increase_labels()
    training_ids = projector.instance_ids_for_split("training")
    test_ids = projector.instance_ids_for_split("test")

    assert labels.shape == (39,)
    assert labels.sum() == 25
    assert training_ids == tuple(range(10))
    assert test_ids == tuple(range(10, 39))
    assert projector.increase_labels(training_ids).tolist() == [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]


def test_original_prediction_is_checked_against_explanation(projector):
    deltas = projector.deltas.copy()
    deltas.loc[deltas["instanceId"] == 1, "originalPrediction"] = 1
    inconsistent = Sim2RealAttributionProjector(
        projector.values,
        projector.attributions,
        projector.counterfactuals,
        metadata=projector.metadata,
        deltas=deltas,
    )

    with pytest.raises(ValueError, match="originalPrediction=1.*pred=0"):
        inconsistent.project(1, exp_property="faithful")


def test_fits_increase_response_directly_from_projector(projector):
    model = Sim2RealFittedAttributionSum(aggregation="value_weighted")
    returned = model.fit_from_projector(
        projector,
        exp_property="robust",
        split=None,
    )

    pairs = projector.project_all(exp_property="robust")
    probabilities = model.predict_proba(pairs)

    assert returned is model
    assert model.is_fitted
    assert model.fit_instance_ids_ == tuple(range(39))
    assert model.comparison_scale > 0.0
    assert model.training_accuracy_ == pytest.approx(1.0)
    assert probabilities.shape == (39, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_fit_defaults_to_raw_training_split(projector):
    model = Sim2RealFittedAttributionSum(missing_attributions="zero")
    model.fit_from_projector(
        projector,
        exp_property="sparse_robust",
    )

    assert model.fit_instance_ids_ == tuple(range(10))
    assert model.training_accuracy_ is not None


def test_comparison_regularization_is_configurable(projector):
    pairs = projector.project_all(
        exp_property="robust",
        instance_ids=projector.instance_ids_for_split("training"),
    )
    model = Sim2RealFittedAttributionSum(comparison_C=0.1).fit(pairs)

    assert model.comparison_C == pytest.approx(0.1)
    assert model.comparison_estimator_.C == pytest.approx(0.1)

    with pytest.raises(ValueError, match="comparison_C"):
        Sim2RealFittedAttributionSum(comparison_C=0)


def test_projector_and_model_can_load_repository_assets_directly():
    projector = Sim2RealAttributionProjector.from_assets()
    pair = projector.project(1, exp_property="faithful")
    model = Sim2RealFittedAttributionSum(
        aggregation="value_weighted"
    ).fit_from_assets(exp_property="robust")

    assert pair.changed_feature_names == ("sex",)
    assert model.fit_instance_ids_ == tuple(range(10))


def test_exemplar_memory_only_retrieves_when_attribution_is_invisible(projector):
    """CoAX bypasses memory whenever the explanation supplies the weight."""
    training = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("training"),
    )
    test = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("test"),
    )
    model = Sim2RealFittedAttributionSum(
        use_exemplar_memory=True,
        memory_sensitivity=2.0,
        max_features_attended=12,
    ).fit(training)

    assert model.memory_exemplars_

    for comparison in model.compare_many(test):
        if comparison.changed_feature_visible:
            assert comparison.retrieved_exemplar_count == 0
            assert comparison.memory_imputed_weight_change is None
        else:
            assert comparison.retrieved_exemplar_count > 0
            assert comparison.memory_imputed_weight_change is not None


def test_exemplar_memory_leaves_visible_trials_identical(projector):
    """Turning memory on must not move a trial whose weight is displayed."""
    training = projector.project_all(
        exp_property="faithful",
        instance_ids=projector.instance_ids_for_split("training"),
    )
    test = projector.project_all(
        exp_property="faithful",
        instance_ids=projector.instance_ids_for_split("test"),
    )
    baseline = Sim2RealFittedAttributionSum().fit(training)
    with_memory = Sim2RealFittedAttributionSum(
        use_exemplar_memory=True, memory_sensitivity=2.0
    ).fit(training)

    visible = [
        (before, after)
        for before, after in zip(
            baseline.compare_many(test), with_memory.compare_many(test)
        )
        if before.changed_feature_visible
    ]

    assert visible
    for before, after in visible:
        assert after.confidence_delta == pytest.approx(before.confidence_delta)


def test_memory_response_still_comes_from_the_confidence_delta(projector):
    """Memory fills the missing weight; p_new - p_original still decides."""
    training = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("training"),
    )
    test = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("test"),
    )
    model = Sim2RealFittedAttributionSum(
        use_exemplar_memory=True,
        memory_sensitivity=2.0,
        max_features_attended=12,
    ).fit(training)

    for comparison in model.compare_many(test):
        assert comparison.confidence_delta == pytest.approx(
            comparison.p_new_high_income - comparison.p_original_high_income
        )
        expected = 1.0 / (
            1.0
            + math.exp(
                -(
                    model.comparison_intercept
                    + model.comparison_scale * comparison.confidence_delta
                )
            )
        )
        assert comparison.pre_lapse_probability_income_increases == pytest.approx(
            expected
        )


def test_retrieval_threshold_can_silence_the_memory(projector):
    training = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("training"),
    )
    test = projector.project_all(
        exp_property="sparse",
        instance_ids=projector.instance_ids_for_split("test"),
    )
    model = Sim2RealFittedAttributionSum(
        use_exemplar_memory=True,
        retrieval_threshold=1e6,
        max_features_attended=12,
    ).fit(training)

    for comparison in model.compare_many(test):
        assert comparison.retrieved_exemplar_count == 0
        if not comparison.changed_feature_visible:
            assert comparison.memory_imputed_weight_change == pytest.approx(0.0)

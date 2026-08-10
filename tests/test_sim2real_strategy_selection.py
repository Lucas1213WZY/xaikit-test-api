"""Three strategies compete on one grid, and BIC has to price them fairly.

``Sim2RealFittedAttributionSum`` reads the explanation, so it answers at chance
on every trial whose changed feature has no visible attribution -- 93% of the
``sparse`` condition. The two exemplar strategies read feature values instead.
Letting them compete per participant only means anything if the candidate space
and the parameter count treat them on equal terms, which is what these tests
pin: a candidate must be built as the strategy it names, must not be charged for
parameters its own model cannot read, and must not multiply the grid with
variants of fields it ignores.
"""

from dataclasses import asdict, replace

import pytest

from src.cognitive_models.cognitive_models.fit_sim2real_attribution_sum_to_participants import (
    ATTRIBUTION_ONLY_FIELDS,
    GCM_ONLY_FIELDS,
    GCM_STRATEGIES,
    MEMORY_ONLY_FIELDS,
    STRATEGY_CLASSES,
    AttributionSumCandidate,
    TrainedCandidateEvaluator,
    candidate_grid,
    count_free_parameters,
)
from src.cognitive_models.cognitive_models.Sim2Real import (
    Sim2RealAttributionProjector,
    Sim2RealFittedAttributionSum,
    Sim2RealFittedSalientFeatures,
    Sim2RealFittedSensitiveFeatures,
)

ALL_STRATEGIES = ("attribution_sum", "sensitive_features", "salient_features")


@pytest.fixture(scope="module")
def projector():
    return Sim2RealAttributionProjector.from_assets()


@pytest.fixture(scope="module")
def evaluator(projector):
    return TrainedCandidateEvaluator(projector, missing_attributions="zero")


def _candidate(strategy: str, **overrides) -> AttributionSumCandidate:
    base = AttributionSumCandidate(
        aggregation="attribution",
        normalize_by_i_max=False,
        confidence_scale=1.0,
        confidence_intercept=0.0,
        comparison_C=1e6,
        strategy=strategy,
    )
    return replace(base, **overrides)


# -- _build_model dispatch -------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("attribution_sum", Sim2RealFittedAttributionSum),
        ("sensitive_features", Sim2RealFittedSensitiveFeatures),
        ("salient_features", Sim2RealFittedSalientFeatures),
    ],
)
def test_a_candidate_is_built_as_the_strategy_it_names(evaluator, strategy, expected):
    assert isinstance(evaluator._build_model(_candidate(strategy)), expected)


def test_the_registry_and_the_gcm_list_agree():
    assert set(GCM_STRATEGIES) < set(STRATEGY_CLASSES)
    assert set(STRATEGY_CLASSES) == set(ALL_STRATEGIES)


def test_gcm_parameters_reach_the_gcm_model(evaluator):
    model = evaluator._build_model(
        _candidate("sensitive_features", k=5, sensitivity=20.0, always_attend_changed=True)
    )
    assert model.k == 5
    assert model.sensitivity == 20.0
    assert model.always_attend_changed is True


def test_attribution_parameters_reach_the_attribution_model(evaluator):
    model = evaluator._build_model(
        _candidate(
            "attribution_sum",
            aggregation="value_weighted",
            confidence_scale=8.0,
            max_features_attended=4,
        )
    )
    assert model.aggregation == "value_weighted"
    assert model.confidence_scale == 8.0
    assert model.max_features_attended == 4


def test_an_unknown_strategy_is_rejected_rather_than_silently_built(evaluator):
    with pytest.raises(ValueError, match="Unknown strategy"):
        evaluator._build_model(_candidate("telepathy"))


def test_every_strategy_trains_and_infers(evaluator):
    """The dispatch is only useful if each strategy survives a real fit."""
    for strategy in ALL_STRATEGIES:
        inference = evaluator.evaluate("sparse", _candidate(strategy))
        assert len(inference.test_predictions) == 29
        assert inference.test_predictions["model_probability_higher"].notna().all()


def test_the_winning_strategy_is_recorded_in_the_fit_output():
    """participant_fits.csv is written from asdict(candidate)."""
    assert "strategy" in asdict(_candidate("sensitive_features"))


# -- parameter charging ----------------------------------------------------


def test_a_gcm_candidate_is_not_charged_for_attribution_parameters():
    """It cannot read them, so they are not free parameters of that model."""
    candidates = candidate_grid(strategies=ALL_STRATEGIES)
    gcm = next(item for item in candidates if item.strategy == "sensitive_features")
    attribution = next(
        item for item in candidates if item.strategy == "attribution_sum"
    )
    assert count_free_parameters(gcm, candidates) < count_free_parameters(
        attribution, candidates
    )


def test_an_attribution_candidate_is_not_charged_for_gcm_parameters():
    candidates = candidate_grid(strategies=ALL_STRATEGIES, ks=(2, 8))
    attribution = next(
        item for item in candidates if item.strategy == "attribution_sum"
    )
    without_gcm_variation = candidate_grid(strategies=ALL_STRATEGIES, ks=(2,))
    twin = next(
        item for item in without_gcm_variation if item.strategy == "attribution_sum"
    )
    assert count_free_parameters(attribution, candidates) == count_free_parameters(
        twin, without_gcm_variation
    )


def test_the_two_field_groups_do_not_overlap():
    assert not set(ATTRIBUTION_ONLY_FIELDS) & set(GCM_ONLY_FIELDS)
    assert set(MEMORY_ONLY_FIELDS) <= set(ATTRIBUTION_ONLY_FIELDS)


def test_memory_fields_are_still_only_charged_when_memory_is_on():
    """The pre-existing rule must survive the per-strategy rule."""
    candidates = candidate_grid(
        strategies=("attribution_sum",),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        comparison_cs=(1e6,),
        aggregations=("attribution",),
        normalize_options=(False,),
        max_features_attended_options=(12,),
        memory_options=(False, True),
        memory_sensitivities=(1.0, 2.0),
    )
    off = next(item for item in candidates if not item.use_exemplar_memory)
    on = next(item for item in candidates if item.use_exemplar_memory)
    assert count_free_parameters(on, candidates) > count_free_parameters(off, candidates)


# -- grid composition ------------------------------------------------------


def test_each_requested_strategy_appears():
    candidates = candidate_grid(strategies=ALL_STRATEGIES)
    assert {item.strategy for item in candidates} == set(ALL_STRATEGIES)


def test_the_default_grid_is_attribution_only():
    """Adding strategies must be opt-in, so existing runs are unchanged."""
    assert {item.strategy for item in candidate_grid()} == {"attribution_sum"}


def test_excluding_attribution_sum_leaves_only_the_exemplar_strategies():
    candidates = candidate_grid(strategies=("sensitive_features",))
    assert {item.strategy for item in candidates} == {"sensitive_features"}


def test_gcm_candidates_pin_the_fields_they_ignore():
    """Otherwise the attribution axes would multiply the space with duplicates."""
    candidates = [
        item
        for item in candidate_grid(
            strategies=("sensitive_features",),
            aggregations=("attribution", "value_weighted"),
            normalize_options=(False, True),
            confidence_scales=(0.5, 2.0, 8.0),
        )
    ]
    for field in ("aggregation", "normalize_by_i_max", "confidence_scale"):
        assert len({getattr(item, field) for item in candidates}) == 1, field


def test_the_grid_holds_no_duplicates():
    candidates = candidate_grid(strategies=ALL_STRATEGIES)
    assert len(set(candidates)) == len(candidates)


def test_gcm_axes_do_not_inflate_the_attribution_space():
    """k and sensitivity belong to the exemplar strategies only."""
    narrow = candidate_grid(strategies=("attribution_sum",), ks=(2,), sensitivities=(1.0,))
    wide = candidate_grid(
        strategies=("attribution_sum",), ks=(2, 3, 5, 8), sensitivities=(1.0, 4.0)
    )
    assert len(narrow) == len(wide)


def test_an_unknown_strategy_is_rejected_by_the_grid():
    with pytest.raises(ValueError, match="Unknown strategy"):
        candidate_grid(strategies=("attribution_sum", "telepathy"))


def test_shared_parameters_still_sweep_across_every_strategy():
    """guess_bias/lapse_rate/comparison_C are read by all three, so they must
    keep varying -- otherwise the strategies are not compared on equal terms."""
    candidates = candidate_grid(
        strategies=ALL_STRATEGIES, comparison_cs=(1e2, 1e8), lapse_rates=(0.0, 0.5)
    )
    for strategy in ALL_STRATEGIES:
        subset = [item for item in candidates if item.strategy == strategy]
        assert len({item.comparison_C for item in subset}) == 2, strategy
        assert len({item.lapse_rate for item in subset}) == 2, strategy


def test_attribution_axes_do_not_inflate_the_gcm_space():
    """The count of exemplar candidates must not scale with attribution axes.

    Pinning the ignored fields is what keeps the grid from multiplying: without
    it, widening ``aggregations`` or ``confidence_scales`` would produce N
    identical exemplar candidates that differ only in fields their model never
    reads, inflating both the search cost and the parameter count.
    """
    narrow = candidate_grid(
        strategies=("sensitive_features",),
        aggregations=("attribution",),
        normalize_options=(False,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
    )
    wide = candidate_grid(
        strategies=("sensitive_features",),
        aggregations=("attribution", "value_weighted"),
        normalize_options=(False, True),
        confidence_scales=(0.5, 2.0, 8.0, 32.0),
        confidence_intercepts=(-1.0, 0.0, 1.0),
        max_features_attended_options=(4, 12),
    )
    assert len(narrow) == len(wide), (
        f"exemplar candidates grew from {len(narrow)} to {len(wide)} when only "
        "attribution axes widened"
    )

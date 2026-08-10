import json

import pandas as pd
import pytest

from src.cognitive_models.cognitive_models.fit_sim2real_attribution_sum_to_participants import (
    AttributionSumCandidate,
    TrainedCandidateEvaluator,
    attach_test_instances,
    load_screened_participant_trials,
    run_fitting,
    candidate_grid,
    condition_specific_candidate_grids,
    count_free_parameters,
    _binary_metrics,
)
from src.cognitive_models.cognitive_models.sim2real.gcm_strategies import (
    Sim2RealAttributionProjector,
)


def _participant_rows(projector, *, explanation_type=2, missing_qid=None):
    deltas = projector.deltas[projector.deltas["split"] == "test"].sort_values(
        "qid"
    )
    rows = []
    for trial_index, row in enumerate(deltas.itertuples(), start=10):
        if row.qid == missing_qid:
            continue
        answer = "Higher" if row.answer == 1 else "Lower"
        rows.append(
            {
                "trial_index": trial_index,
                "trial_type": "survey-multi-choice",
                "question_type": "our_prop_good_under_model",
                "qid": row.qid,
                "explanation_type": explanation_type,
                "responses": json.dumps({"Q0": answer}),
                "answer": answer,
                "correct": True,
            }
        )
    return pd.DataFrame(rows)


def test_screening_keeps_only_complete_qid_sets(tmp_path):
    projector = Sim2RealAttributionProjector.from_assets()
    _participant_rows(projector).to_csv(
        tmp_path / "clinical_complete.csv", index=False
    )
    _participant_rows(projector, missing_qid=28).to_csv(
        tmp_path / "clinical_incomplete.csv", index=False
    )

    trials, screening = load_screened_participant_trials(tmp_path)

    assert trials["participant_id"].unique().tolist() == ["complete"]
    assert len(trials) == 29
    assert screening.set_index("participant_id").loc["complete", "eligible"]
    assert not screening.set_index("participant_id").loc[
        "incomplete", "eligible"
    ]


def test_candidate_is_trained_on_training_then_infers_test():
    projector = Sim2RealAttributionProjector.from_assets()
    evaluator = TrainedCandidateEvaluator(projector)
    candidate = AttributionSumCandidate(
        aggregation="value_weighted",
        normalize_by_i_max=False,
        confidence_scale=1.0,
        confidence_intercept=0.0,
        comparison_C=1.0,
    )

    inference = evaluator.evaluate("robust", candidate)

    assert inference.training_accuracy >= 0.5
    assert inference.training_instance_ids == tuple(range(10))
    assert inference.test_predictions["qid"].tolist() == list(range(29))
    assert inference.test_predictions["instanceId"].tolist() == list(range(10, 39))


def test_candidate_grid_uses_requested_attention_limits():
    candidates = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        comparison_cs=(1.0,),
        max_features_attended_options=(1, 2, 4, 6, 8, 12),
        memory_options=(False,),
    )

    assert [item.max_features_attended for item in candidates] == [1, 2, 4, 6, 8, 12]


def test_candidate_grid_exposes_guess_bias_and_lapse_rate():
    candidates = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        comparison_cs=(1.0,),
        max_features_attended_options=(12,),
        guess_biases=(-1.0, 0.0, 1.0),
        lapse_rates=(0.0, 0.2),
        memory_options=(False,),
    )

    assert [(item.guess_bias, item.lapse_rate) for item in candidates] == [
        (-1.0, 0.0),
        (-1.0, 0.2),
        (0.0, 0.0),
        (0.0, 0.2),
        (1.0, 0.0),
        (1.0, 0.2),
    ]


def test_run_fitting_writes_screened_train_then_test_results(tmp_path):
    projector = Sim2RealAttributionProjector.from_assets()
    participant_dir = tmp_path / "participants"
    output_dir = tmp_path / "results"
    participant_dir.mkdir()
    _participant_rows(projector).to_csv(
        participant_dir / "clinical_complete.csv", index=False
    )
    _participant_rows(projector, missing_qid=0).to_csv(
        participant_dir / "clinical_screened.csv", index=False
    )
    candidate = AttributionSumCandidate(
        aggregation="attribution",
        normalize_by_i_max=False,
        confidence_scale=1.0,
        confidence_intercept=0.0,
        comparison_C=1.0,
    )

    outputs = run_fitting(
        participant_dir=participant_dir,
        output_dir=output_dir,
        candidates=[candidate],
    )

    assert len(outputs["participant_fits"]) == 1
    assert len(outputs["participant_predictions"]) == 29
    assert outputs["participant_fits"].iloc[0]["n_trials"] == 29
    assert int(outputs["screening"]["eligible"].sum()) == 1
    assert (output_dir / "run_config.json").is_file()
    assert (output_dir / "participant_fits.csv").is_file()
    assert "prolific_id" not in outputs["participant_predictions"].columns
    assert outputs["participant_predictions"]["guess_bias"].eq(0.0).all()
    assert outputs["participant_predictions"]["lapse_rate"].eq(0.0).all()
    assert "pre_lapse_probability_higher" in outputs[
        "participant_predictions"
    ].columns


def test_run_fitting_can_restrict_candidates_by_explanation_property(tmp_path):
    projector = Sim2RealAttributionProjector.from_assets()
    participant_dir = tmp_path / "participants"
    output_dir = tmp_path / "results"
    participant_dir.mkdir()
    _participant_rows(projector, explanation_type=0).to_csv(
        participant_dir / "clinical_faithful.csv", index=False
    )
    _participant_rows(projector, explanation_type=1).to_csv(
        participant_dir / "clinical_sparse.csv", index=False
    )
    faithful_candidate = AttributionSumCandidate(
        aggregation="attribution",
        normalize_by_i_max=False,
        confidence_scale=1.0,
        confidence_intercept=0.0,
        comparison_C=1.0,
        guess_bias=0.0,
        lapse_rate=0.0,
    )
    sparse_candidate = AttributionSumCandidate(
        aggregation="attribution",
        normalize_by_i_max=False,
        confidence_scale=1.0,
        confidence_intercept=0.0,
        comparison_C=1.0,
        guess_bias=2.0,
        lapse_rate=1.0,
    )

    outputs = run_fitting(
        participant_dir=participant_dir,
        output_dir=output_dir,
        candidates=[faithful_candidate, sparse_candidate],
        candidates_by_property={
            "faithful": [faithful_candidate],
            "sparse": [sparse_candidate],
        },
    )

    fits = outputs["participant_fits"].set_index("exp_property")
    assert fits.loc["faithful", "guess_bias"] == 0.0
    assert fits.loc["faithful", "lapse_rate"] == 0.0
    assert fits.loc["sparse", "guess_bias"] == 2.0
    assert fits.loc["sparse", "lapse_rate"] == 1.0


def test_candidate_grid_collapses_duplicate_memory_off_candidates():
    """Memory settings must not multiply candidates that never use memory."""
    candidates = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        comparison_cs=(1e6,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
        memory_options=(False, True),
        memory_sensitivities=(1.0, 2.0, 4.0),
    )

    off = [item for item in candidates if not item.use_exemplar_memory]
    on = [item for item in candidates if item.use_exemplar_memory]

    assert len(off) == 1
    assert len(on) == 3
    assert len(set(candidates)) == len(candidates)


def test_memory_only_parameters_are_charged_only_when_memory_is_on():
    candidates = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        comparison_cs=(1e6,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
        memory_options=(False, True),
        memory_sensitivities=(1.0, 2.0, 4.0),
    )
    off = next(item for item in candidates if not item.use_exemplar_memory)
    on = next(item for item in candidates if item.use_exemplar_memory)

    # comparison_scale + comparison_intercept, plus use_exemplar_memory, which
    # varies across the space. Only the memory candidate also pays for
    # memory_sensitivity.
    assert count_free_parameters(off, candidates) == 3
    assert count_free_parameters(on, candidates) == 4


def test_pinned_parameters_are_not_charged():
    candidates = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        comparison_cs=(1e6,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
        memory_options=(False,),
    )

    assert len(candidates) == 1
    assert count_free_parameters(candidates[0], candidates) == 2


def test_bic_is_the_standard_penalised_likelihood():
    """k*ln(n) - 2*LL. Not inherited from CoAX -- that repo ships fitted
    outputs but no fitting code, and CoXAM's notebook selects on NLL.
    """
    import numpy as np

    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.3, 0.6, 0.4])
    metrics = _binary_metrics(labels, probabilities, n_parameters=3)

    expected_total = -float(
        np.sum(
            labels * np.log(probabilities)
            + (1 - labels) * np.log(1 - probabilities)
        )
    )
    assert metrics["total_nll"] == pytest.approx(expected_total)
    assert metrics["nll"] == pytest.approx(expected_total / 4)
    assert metrics["bic"] == pytest.approx(
        3 * np.log(4) + 2 * expected_total
    )


def test_condition_specific_grids_restrict_guessing_to_sparse():
    grids = condition_specific_candidate_grids(
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
        memory_options=(False,),
    )

    for exp_property in ("faithful", "robust", "sparse_robust"):
        assert {item.lapse_rate for item in grids[exp_property]} == {0.0}
        assert {item.guess_bias for item in grids[exp_property]} == {0.0}

    assert {item.lapse_rate for item in grids["sparse"]} == {
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert {item.guess_bias for item in grids["sparse"]} == {0.0, 1.0, 2.0}


def test_lapse_variants_reuse_the_base_fit_exactly():
    """The shared-base shortcut must match a directly fitted lapse model."""
    from src.cognitive_models.cognitive_models.sim2real.gcm_strategies import (
        Sim2RealFittedAttributionSum,
    )

    projector = Sim2RealAttributionProjector.from_assets()
    evaluator = TrainedCandidateEvaluator(projector, missing_attributions="zero")
    candidate = AttributionSumCandidate(
        aggregation="value_weighted",
        normalize_by_i_max=True,
        confidence_scale=2.0,
        confidence_intercept=1.0,
        comparison_C=1e6,
        max_features_attended=4,
        guess_bias=1.0,
        lapse_rate=0.5,
    )

    inference = evaluator.evaluate("sparse", candidate)

    direct = Sim2RealFittedAttributionSum(
        confidence_scale=candidate.confidence_scale,
        confidence_intercept=candidate.confidence_intercept,
        comparison_C=candidate.comparison_C,
        guess_bias=candidate.guess_bias,
        lapse_rate=candidate.lapse_rate,
        aggregation=candidate.aggregation,
        max_features_attended=candidate.max_features_attended,
        missing_attributions="zero",
    )
    training = projector.project_all(
        exp_property="sparse",
        normalize_by_i_max=True,
        instance_ids=projector.instance_ids_for_split("training"),
    )
    test = projector.project_all(
        exp_property="sparse",
        normalize_by_i_max=True,
        instance_ids=projector.instance_ids_for_split("test"),
    )
    direct.fit(training)
    expected = direct.predict_proba(test)[:, 1]

    assert inference.test_predictions["model_probability_higher"].to_numpy() == (
        pytest.approx(expected)
    )
    assert inference.comparison_scale == pytest.approx(direct.comparison_scale)


def test_goodness_of_fit_leads_the_selection_key():
    """NLL decides, because the search is the maximum-likelihood estimation.

    BIC is defined as -2 ln(L-hat) + k ln(n) against the *maximised*
    likelihood, so it cannot also be what picks the parameterization. It is
    computed from the winning fit and reported for comparing whole models
    (different strategies) on the same data.
    """
    from src.cognitive_models.cognitive_models.fit_sim2real_attribution_sum_to_participants import (
        _candidate_sort_key,
        candidate_grid,
    )

    candidate = candidate_grid(
        aggregations=("attribution",),
        normalize_options=(False,),
        comparison_cs=(1e6,),
        confidence_scales=(1.0,),
        confidence_intercepts=(0.0,),
        max_features_attended_options=(12,),
        memory_options=(False,),
    )[0]

    base = {"nll": 0.50, "bic": 100.0, "accuracy": 0.7, "brier": 0.2}
    better_fit_pricier = {**base, "nll": 0.40, "bic": 200.0}
    worse_fit_cheaper = {**base, "nll": 0.60, "bic": 10.0}
    tie_on_fit_cheaper = {**base, "bic": 10.0}

    assert _candidate_sort_key(better_fit_pricier, candidate) < _candidate_sort_key(
        worse_fit_cheaper, candidate
    ), "a better-fitting candidate must win even when its BIC is worse"
    assert _candidate_sort_key(tie_on_fit_cheaper, candidate) < _candidate_sort_key(
        base, candidate
    ), "on an NLL tie the lower-BIC candidate must win"

"""Fit the Sim2Real attribution-sum cognitive model to participant responses.

The fitting sequence mirrors the study rather than training directly on human
test answers:

1. Screen participant files to those containing exactly test qids 0..28.
2. Select the explanation property from each participant's condition.
3. Train each candidate cognitive model on the ten generated training cases
   and their Higher/Lower feedback from ``deltas.csv``.
4. Freeze the trained model and infer the 29 testing cases.
5. Select candidate parameters by model-versus-participant negative
   log-likelihood on ``responses.Q0``.

The source participant files and Sim2Real assets are read-only. Results are
written to an explicitly selected output directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .sim2real_fitted_attribution_sum import (
    REPO_ROOT,
    Sim2RealAttributionProjector,
    Sim2RealFittedAttributionSum,
)


EXPECTED_TEST_QIDS = frozenset(range(29))
EXP_PROPERTY_BY_EXPLANATION_TYPE = {
    0: "faithful",
    1: "sparse",
    2: "robust",
    3: "sparse_robust",
}
HUMAN_TEST_QUESTION_TYPE = "our_prop_good_under_model"


@dataclass(frozen=True)
class AttributionSumCandidate:
    """One discrete/continuous cognitive-model parameter candidate."""

    aggregation: str
    normalize_by_i_max: bool
    confidence_scale: float
    confidence_intercept: float
    comparison_C: float
    max_features_attended: int = 12
    guess_bias: float = 0.0
    lapse_rate: float = 0.0
    use_exemplar_memory: bool = False
    memory_sensitivity: float = 1.0
    memory_decay: float = 0.5
    retrieval_threshold: Optional[float] = None


#: Fields that only affect a candidate when its exemplar memory is switched on.
MEMORY_ONLY_FIELDS = ("memory_sensitivity", "memory_decay", "retrieval_threshold")

#: Parameters the training-case logistic learns for every candidate.
LEARNED_PARAMETER_NAMES = ("comparison_scale", "comparison_intercept")


@dataclass(frozen=True)
class CandidateInference:
    """A candidate after training-case feedback and test inference."""

    candidate: AttributionSumCandidate
    exp_property: str
    comparison_scale: float
    comparison_intercept: float
    training_accuracy: float
    training_instance_ids: tuple[int, ...]
    test_predictions: pd.DataFrame


def _parse_human_answer(value, *, participant_id: str, qid: int) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"participant={participant_id} qid={qid} has invalid responses JSON"
        ) from exc
    answer = payload.get("Q0")
    if answer not in {"Higher", "Lower"}:
        raise ValueError(
            f"participant={participant_id} qid={qid} has Q0={answer!r}; "
            "expected Higher or Lower"
        )
    return answer


def load_screened_participant_trials(
    participant_dir,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only participants with exactly one response for every test qid."""
    participant_dir = Path(participant_dir)
    files = sorted(participant_dir.glob("clinical_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No clinical_*.csv participant files found in {participant_dir}"
        )

    eligible_frames: list[pd.DataFrame] = []
    screening_rows: list[dict] = []
    for path in files:
        participant_id = path.stem.removeprefix("clinical_")
        frame = pd.read_csv(path)
        if "question_type" in frame.columns:
            test = frame[
                frame["question_type"].eq(HUMAN_TEST_QUESTION_TYPE)
            ].copy()
        else:
            test = frame.iloc[0:0].copy()

        if "qid" not in test.columns:
            test["qid"] = pd.Series(dtype=float)
        numeric_qids = pd.to_numeric(test["qid"], errors="coerce")
        valid_qids = numeric_qids.dropna().astype(int)
        qid_set = set(valid_qids.tolist())
        duplicated_qids = sorted(
            valid_qids[valid_qids.duplicated(keep=False)].unique().tolist()
        )
        missing_qids = sorted(EXPECTED_TEST_QIDS.difference(qid_set))
        unexpected_qids = sorted(qid_set.difference(EXPECTED_TEST_QIDS))
        complete = (
            len(test) == len(EXPECTED_TEST_QIDS)
            and len(valid_qids) == len(EXPECTED_TEST_QIDS)
            and not duplicated_qids
            and not missing_qids
            and not unexpected_qids
        )

        explanation_types = (
            pd.to_numeric(test.get("explanation_type"), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "explanation_type" in test.columns
            else []
        )
        explanation_type = explanation_types[0] if len(explanation_types) == 1 else None
        exp_property = EXP_PROPERTY_BY_EXPLANATION_TYPE.get(explanation_type)
        if complete and exp_property is None:
            complete = False

        reason = "eligible"
        if not complete:
            if missing_qids or unexpected_qids or duplicated_qids:
                reason = "incomplete_or_invalid_test_qids"
            elif exp_property is None:
                reason = "invalid_explanation_type"
            else:
                reason = "missing_test_rows"

        screening_rows.append(
            {
                "participant_id": participant_id,
                "source_file": path.name,
                "eligible": complete,
                "reason": reason,
                "n_test_rows": len(test),
                "n_unique_test_qids": len(qid_set),
                "missing_qids": ";".join(map(str, missing_qids)),
                "unexpected_qids": ";".join(map(str, unexpected_qids)),
                "duplicate_qids": ";".join(map(str, duplicated_qids)),
                "explanation_type": explanation_type,
                "exp_property": exp_property,
            }
        )
        if not complete:
            continue

        test["qid"] = valid_qids.to_numpy()
        test["participant_id"] = participant_id
        test["source_file"] = path.name
        test["explanation_type"] = explanation_type
        test["exp_property"] = exp_property
        test["human_answer"] = [
            _parse_human_answer(
                value,
                participant_id=participant_id,
                qid=int(qid),
            )
            for value, qid in zip(test["responses"], test["qid"])
        ]
        test["human_label"] = test["human_answer"].map(
            {"Lower": 0, "Higher": 1}
        )
        eligible_frames.append(test)

    screening = pd.DataFrame(screening_rows)
    if not eligible_frames:
        raise ValueError("No participants passed the complete-qid screening rule")
    trials = pd.concat(eligible_frames, ignore_index=True, sort=False)
    return trials, screening


def attach_test_instances(
    trials: pd.DataFrame,
    projector: Sim2RealAttributionProjector,
) -> pd.DataFrame:
    """Join testing qids to generated instance IDs and validate answer keys."""
    deltas = projector.deltas
    if deltas is None:
        raise ValueError("deltas.csv is required for qid-to-instance mapping")
    mapping = deltas[
        deltas["appId"].eq(projector.app_id)
        & deltas["split"].astype(str).eq("test")
    ][["qid", "instanceId", "answer"]].copy()
    mapping["qid"] = pd.to_numeric(mapping["qid"], errors="raise").astype(int)
    if mapping["qid"].duplicated().any():
        raise ValueError("Testing deltas contain duplicate qids")
    if set(mapping["qid"]) != EXPECTED_TEST_QIDS:
        raise ValueError("Testing deltas must contain exactly qids 0..28")
    mapping = mapping.rename(
        columns={"answer": "correct_label"}
    )
    mapping["correct_answer"] = mapping["correct_label"].map(
        {0: "Lower", 1: "Higher"}
    )

    attached = trials.merge(mapping, on="qid", how="left", validate="many_to_one")
    if attached["instanceId"].isna().any():
        raise ValueError("Some participant qids did not map to testing instances")
    attached["instanceId"] = attached["instanceId"].astype(int)
    if "answer" in attached.columns:
        mismatch = attached["answer"].notna() & ~attached["answer"].eq(
            attached["correct_answer"]
        )
        if mismatch.any():
            row = attached[mismatch].iloc[0]
            raise ValueError(
                f"participant={row['participant_id']} qid={row['qid']} has "
                f"answer={row['answer']!r}, expected {row['correct_answer']!r}"
            )
    return attached


def candidate_grid(
    *,
    aggregations: Sequence[str] = ("attribution",),
    normalize_options: Sequence[bool] = (False,),
    confidence_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    confidence_intercepts: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    comparison_cs: Sequence[float] = (1e6,),
    max_features_attended_options: Sequence[int] = (4, 12),
    guess_biases: Sequence[float] = (0.0,),
    lapse_rates: Sequence[float] = (0.0,),
    memory_options: Sequence[bool] = (False, True),
    memory_sensitivities: Sequence[float] = (1.0, 2.0, 4.0),
    memory_decays: Sequence[float] = (0.5,),
    retrieval_thresholds: Sequence[Optional[float]] = (None,),
) -> list[AttributionSumCandidate]:
    """Construct the deterministic candidate space used for fitting.

    The defaults are the trimmed space: ``aggregation``, ``normalize_by_i_max``
    and ``comparison_C`` are pinned because a full sweep showed they never buy
    enough negative log likelihood to pay for themselves under BIC, while
    ``max_features_attended`` is kept at two values because it also decides
    which features get stored as exemplars.  Candidates whose memory is off are
    collapsed onto the default memory settings so the grid holds no duplicates.
    """
    candidates = [
        AttributionSumCandidate(
            aggregation=aggregation,
            normalize_by_i_max=bool(normalize),
            confidence_scale=float(confidence_scale),
            confidence_intercept=float(confidence_intercept),
            comparison_C=float(comparison_c),
            max_features_attended=int(max_features_attended),
            guess_bias=float(guess_bias),
            lapse_rate=float(lapse_rate),
            use_exemplar_memory=bool(use_memory),
            memory_sensitivity=float(memory_sensitivity) if use_memory else 1.0,
            memory_decay=float(memory_decay) if use_memory else 0.5,
            retrieval_threshold=(
                None
                if not use_memory or retrieval_threshold is None
                else float(retrieval_threshold)
            ),
        )
        for aggregation, normalize, confidence_scale, confidence_intercept, comparison_c, max_features_attended, guess_bias, lapse_rate, use_memory, memory_sensitivity, memory_decay, retrieval_threshold
        in product(
            aggregations,
            normalize_options,
            confidence_scales,
            confidence_intercepts,
            comparison_cs,
            max_features_attended_options,
            guess_biases,
            lapse_rates,
            memory_options,
            memory_sensitivities,
            memory_decays,
            retrieval_thresholds,
        )
    ]
    return list(dict.fromkeys(candidates))


def condition_specific_candidate_grids(
    **grid_options,
) -> dict[str, list[AttributionSumCandidate]]:
    """Per-condition candidate spaces with guessing restricted to sparse.

    Faithful, robust and sparse+robust participants are not in an
    "explanation carries no usable information" condition, so their candidates
    keep ``guess_bias=0`` and ``lapse_rate=0``.  Only sparse searches the
    lapse and guessing grid.
    """
    without_guessing = candidate_grid(**grid_options)
    with_guessing = candidate_grid(
        **{
            **grid_options,
            "guess_biases": (0.0, 1.0, 2.0),
            "lapse_rates": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        }
    )
    return {
        "faithful": without_guessing,
        "robust": without_guessing,
        "sparse_robust": without_guessing,
        "sparse": with_guessing,
    }


def count_free_parameters(
    candidate: AttributionSumCandidate,
    candidates: Sequence[AttributionSumCandidate],
) -> int:
    """Count the parameters BIC should charge this candidate for.

    Every candidate pays for the two coefficients the training-case logistic
    learns.  A searched field is only charged when it actually varies inside
    the candidate space, so pinning a field makes it free.  Memory-only fields
    are charged only to candidates that switched memory on.
    """
    varying = {
        field
        for field in asdict(candidate)
        if len({getattr(item, field) for item in candidates}) > 1
    }
    charged = {
        field
        for field in varying
        if field not in MEMORY_ONLY_FIELDS or candidate.use_exemplar_memory
    }
    return len(LEARNED_PARAMETER_NAMES) + len(charged)


class TrainedCandidateEvaluator:
    """Cache train-then-test inference for each property and candidate."""

    def __init__(
        self,
        projector: Sim2RealAttributionProjector,
        *,
        missing_attributions: str = "zero",
    ):
        self.projector = projector
        self.missing_attributions = missing_attributions
        self._pair_cache: dict[tuple[str, bool, str], list] = {}
        self._inference_cache: dict[
            tuple[str, AttributionSumCandidate], CandidateInference
        ] = {}

    def _pairs(self, exp_property: str, normalize: bool, split: str) -> list:
        key = (exp_property, normalize, split)
        if key not in self._pair_cache:
            instance_ids = self.projector.instance_ids_for_split(split)
            self._pair_cache[key] = self.projector.project_all(
                exp_property=exp_property,
                normalize_by_i_max=normalize,
                instance_ids=instance_ids,
            )
        return self._pair_cache[key]

    def evaluate(
        self,
        exp_property: str,
        candidate: AttributionSumCandidate,
    ) -> CandidateInference:
        """Train a candidate and infer the test cases, reusing base fits.

        ``guess_bias`` and ``lapse_rate`` never change what the model learns:
        the training logistic is fitted on ``confidence_delta``, which they do
        not touch, and at test time they only mix the strategy response with a
        guess.  So every candidate that differs only in those two fields shares
        one trained model, and the lapse mixture is applied afterwards.  This
        keeps a lapse sweep from re-fitting the same model once per value.
        """
        key = (exp_property, candidate)
        if key in self._inference_cache:
            return self._inference_cache[key]

        base_candidate = replace(candidate, guess_bias=0.0, lapse_rate=0.0)
        if base_candidate != candidate:
            base = self.evaluate(exp_property, base_candidate)
            inference = _apply_lapse(base, candidate)
            self._inference_cache[key] = inference
            return inference

        model = Sim2RealFittedAttributionSum(
            confidence_scale=candidate.confidence_scale,
            confidence_intercept=candidate.confidence_intercept,
            comparison_C=candidate.comparison_C,
            guess_bias=candidate.guess_bias,
            lapse_rate=candidate.lapse_rate,
            aggregation=candidate.aggregation,
            max_features_attended=candidate.max_features_attended,
            use_exemplar_memory=candidate.use_exemplar_memory,
            memory_sensitivity=candidate.memory_sensitivity,
            memory_decay=candidate.memory_decay,
            retrieval_threshold=candidate.retrieval_threshold,
            missing_attributions=self.missing_attributions,
        )
        training_pairs = self._pairs(
            exp_property,
            candidate.normalize_by_i_max,
            "training",
        )
        model.fit(training_pairs)
        test_pairs = self._pairs(
            exp_property,
            candidate.normalize_by_i_max,
            "test",
        )
        probabilities = model.predict_proba(test_pairs)[:, 1]
        comparisons = model.compare_many(test_pairs)
        predictions = pd.DataFrame(
            {
                "qid": [pair.qid for pair in test_pairs],
                "instanceId": [pair.instance_id for pair in test_pairs],
                "model_probability_higher": probabilities,
                "model_label": (probabilities >= 0.5).astype(int),
                "p_original_high_income": [
                    item.p_original_high_income for item in comparisons
                ],
                "p_new_high_income": [
                    item.p_new_high_income for item in comparisons
                ],
                "confidence_delta": [item.confidence_delta for item in comparisons],
                "pre_lapse_probability_higher": [
                    item.pre_lapse_probability_income_increases
                    for item in comparisons
                ],
                "guess_probability_higher": [
                    item.guess_probability_income_increases
                    for item in comparisons
                ],
                "changed_feature_visible": [
                    item.changed_feature_visible for item in comparisons
                ],
                "effective_lapse_rate": [
                    item.effective_lapse_rate for item in comparisons
                ],
                "retrieved_exemplar_count": [
                    item.retrieved_exemplar_count for item in comparisons
                ],
                "memory_imputed_weight_change": [
                    item.memory_imputed_weight_change for item in comparisons
                ],
            }
        ).sort_values("qid", ignore_index=True)
        inference = CandidateInference(
            candidate=candidate,
            exp_property=exp_property,
            comparison_scale=model.comparison_scale,
            comparison_intercept=model.comparison_intercept,
            training_accuracy=float(model.training_accuracy_),
            training_instance_ids=model.fit_instance_ids_,
            test_predictions=predictions,
        )
        self._inference_cache[key] = inference
        return inference


def _apply_lapse(
    base: CandidateInference,
    candidate: AttributionSumCandidate,
) -> CandidateInference:
    """Re-derive a lapse/guess variant from an already trained base fit.

    Mirrors ``Sim2RealFittedAttributionSum.compare``: the lapse only applies to
    trials whose changed feature has no visible attribution, and mixes the
    strategy response with ``sigmoid(guess_bias)``.
    """
    predictions = base.test_predictions.copy()
    guess_probability = 1.0 / (1.0 + np.exp(-candidate.guess_bias))
    effective_lapse_rate = np.where(
        predictions["changed_feature_visible"].to_numpy(),
        0.0,
        candidate.lapse_rate,
    )
    pre_lapse = predictions["pre_lapse_probability_higher"].to_numpy()
    probabilities = (
        (1.0 - effective_lapse_rate) * pre_lapse
        + effective_lapse_rate * guess_probability
    )
    predictions["guess_probability_higher"] = guess_probability
    predictions["effective_lapse_rate"] = effective_lapse_rate
    predictions["model_probability_higher"] = probabilities
    predictions["model_label"] = (probabilities >= 0.5).astype(int)
    return CandidateInference(
        candidate=candidate,
        exp_property=base.exp_property,
        comparison_scale=base.comparison_scale,
        comparison_intercept=base.comparison_intercept,
        training_accuracy=base.training_accuracy,
        training_instance_ids=base.training_instance_ids,
        test_predictions=predictions,
    )


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_parameters: int = 0,
) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=int)
    predictions = (probabilities >= 0.5).astype(int)
    total_nll = float(-np.sum(
        labels * np.log(probabilities)
        + (1 - labels) * np.log(1 - probabilities)
    ))
    return {
        "nll": total_nll / len(labels),
        "total_nll": total_nll,
        "n_parameters": float(n_parameters),
        # CoAX's evaluator formula: n_parameters * log(n) + 2 * total_nll.
        "bic": float(n_parameters * np.log(len(labels)) + 2.0 * total_nll),
        "accuracy": float(np.mean(predictions == labels)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def _score_candidate(
    rows: pd.DataFrame,
    inference: CandidateInference,
    *,
    n_parameters: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    joined = rows.merge(
        inference.test_predictions,
        on=["qid", "instanceId"],
        how="left",
        validate="many_to_one",
    )
    if joined["model_probability_higher"].isna().any():
        raise ValueError("Candidate inference did not cover every participant qid")
    metrics = _binary_metrics(
        joined["human_label"].to_numpy(),
        joined["model_probability_higher"].to_numpy(),
        n_parameters=n_parameters,
    )
    return metrics, joined


def _candidate_sort_key(
    metrics: dict[str, float],
    candidate: AttributionSumCandidate,
) -> tuple:
    return (
        metrics["bic"],
        metrics["nll"],
        -metrics["accuracy"],
        metrics["brier"],
        candidate.aggregation != "attribution",
        candidate.normalize_by_i_max,
        abs(candidate.confidence_intercept),
        abs(np.log2(candidate.confidence_scale)),
        abs(np.log10(candidate.comparison_C)),
        candidate.lapse_rate,
        abs(candidate.guess_bias),
        candidate.use_exemplar_memory,
        candidate.memory_sensitivity,
        candidate.max_features_attended,
    )


def _candidates_for_property(
    candidates: Sequence[AttributionSumCandidate],
    candidates_by_property: Mapping[str, Sequence[AttributionSumCandidate]] | None,
    exp_property: str,
) -> Sequence[AttributionSumCandidate]:
    if candidates_by_property is None:
        return candidates
    property_candidates = candidates_by_property.get(exp_property, candidates)
    if not property_candidates:
        raise ValueError(f"No candidates available for exp_property={exp_property}")
    return property_candidates


def fit_participants(
    trials: pd.DataFrame,
    evaluator: TrainedCandidateEvaluator,
    candidates: Sequence[AttributionSumCandidate],
    *,
    candidates_by_property: Mapping[str, Sequence[AttributionSumCandidate]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one train-then-infer parameterization per participant."""
    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for participant_id, rows in trials.groupby("participant_id", sort=True):
        exp_property = str(rows["exp_property"].iloc[0])
        best = None
        property_candidates = _candidates_for_property(
            candidates, candidates_by_property, exp_property
        )
        for candidate in property_candidates:
            inference = evaluator.evaluate(exp_property, candidate)
            metrics, predictions = _score_candidate(
                rows,
                inference,
                n_parameters=count_free_parameters(candidate, property_candidates),
            )
            key = _candidate_sort_key(metrics, candidate)
            if best is None or key < best[0]:
                best = (key, candidate, inference, metrics, predictions)

        _, candidate, inference, metrics, predictions = best
        result_rows.append(
            {
                "participant_id": participant_id,
                "explanation_type": int(rows["explanation_type"].iloc[0]),
                "exp_property": exp_property,
                "n_trials": len(rows),
                "n_training_feedback_cases": len(inference.training_instance_ids),
                "training_instance_ids": ";".join(
                    map(str, inference.training_instance_ids)
                ),
                **asdict(candidate),
                "comparison_scale": inference.comparison_scale,
                "comparison_intercept": inference.comparison_intercept,
                "training_feedback_accuracy": inference.training_accuracy,
                "test_nll_model_participant": metrics["nll"],
                "test_total_nll_model_participant": metrics["total_nll"],
                "n_parameters": int(metrics["n_parameters"]),
                "test_bic_model_participant": metrics["bic"],
                "test_accuracy_model_participant": metrics["accuracy"],
                "test_brier_model_participant": metrics["brier"],
                "test_accuracy_participant_answer_key": float(
                    np.mean(rows["human_label"].to_numpy() == rows["correct_label"].to_numpy())
                ),
            }
        )
        predictions = predictions.copy()
        predictions["participant_id"] = participant_id
        predictions["exp_property"] = exp_property
        predictions["aggregation"] = candidate.aggregation
        predictions["normalize_by_i_max"] = candidate.normalize_by_i_max
        predictions["max_features_attended"] = candidate.max_features_attended
        predictions["guess_bias"] = candidate.guess_bias
        predictions["lapse_rate"] = candidate.lapse_rate
        prediction_frames.append(
            predictions[
                [
                    "participant_id",
                    "exp_property",
                    "trial_index",
                    "qid",
                    "instanceId",
                    "human_answer",
                    "human_label",
                    "correct_answer",
                    "correct_label",
                    "model_probability_higher",
                    "model_label",
                    "p_original_high_income",
                    "p_new_high_income",
                    "confidence_delta",
                    "pre_lapse_probability_higher",
                    "guess_probability_higher",
                    "changed_feature_visible",
                    "effective_lapse_rate",
                    "aggregation",
                    "normalize_by_i_max",
                    "max_features_attended",
                    "guess_bias",
                    "lapse_rate",
                ]
            ]
        )
    return pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True)


def fit_conditions(
    trials: pd.DataFrame,
    evaluator: TrainedCandidateEvaluator,
    candidates: Sequence[AttributionSumCandidate],
    *,
    candidates_by_property: Mapping[str, Sequence[AttributionSumCandidate]] | None = None,
) -> pd.DataFrame:
    """Select one shared train-then-infer parameterization per property."""
    result_rows: list[dict] = []
    for exp_property, rows in trials.groupby("exp_property", sort=True):
        best = None
        property_candidates = _candidates_for_property(
            candidates, candidates_by_property, str(exp_property)
        )
        for candidate in property_candidates:
            inference = evaluator.evaluate(str(exp_property), candidate)
            metrics, _ = _score_candidate(
                rows,
                inference,
                n_parameters=count_free_parameters(candidate, property_candidates),
            )
            key = _candidate_sort_key(metrics, candidate)
            if best is None or key < best[0]:
                best = (key, candidate, inference, metrics)
        _, candidate, inference, metrics = best
        result_rows.append(
            {
                "exp_property": exp_property,
                "n_participants": rows["participant_id"].nunique(),
                "n_trials": len(rows),
                "n_training_feedback_cases": len(inference.training_instance_ids),
                "training_instance_ids": ";".join(
                    map(str, inference.training_instance_ids)
                ),
                **asdict(candidate),
                "comparison_scale": inference.comparison_scale,
                "comparison_intercept": inference.comparison_intercept,
                "training_feedback_accuracy": inference.training_accuracy,
                "test_nll_model_participant": metrics["nll"],
                "test_total_nll_model_participant": metrics["total_nll"],
                "n_parameters": int(metrics["n_parameters"]),
                "test_bic_model_participant": metrics["bic"],
                "test_accuracy_model_participant": metrics["accuracy"],
                "test_brier_model_participant": metrics["brier"],
            }
        )
    return pd.DataFrame(result_rows)


def run_fitting(
    *,
    participant_dir,
    output_dir,
    assets_root=None,
    candidates: Sequence[AttributionSumCandidate] | None = None,
    candidates_by_property: Mapping[str, Sequence[AttributionSumCandidate]] | None = None,
    exp_properties: Sequence[str] | None = None,
    missing_attributions: str = "zero",
) -> dict[str, pd.DataFrame]:
    """Run screening, train-then-test fitting, and save reproducible outputs."""
    trials, screening = load_screened_participant_trials(participant_dir)
    projector = Sim2RealAttributionProjector.from_assets(assets_root=assets_root)
    trials = attach_test_instances(trials, projector)
    if exp_properties is not None:
        requested = tuple(exp_properties)
        unknown = set(requested).difference(EXP_PROPERTY_BY_EXPLANATION_TYPE.values())
        if unknown:
            raise ValueError(f"Unknown exp_property values: {sorted(unknown)}")
        trials = trials[trials["exp_property"].isin(requested)].copy()
        if trials.empty:
            raise ValueError(
                f"No eligible participants in conditions {list(requested)}"
            )
    candidates = list(candidates or candidate_grid())
    if not candidates:
        raise ValueError("At least one candidate parameterization is required")
    evaluator = TrainedCandidateEvaluator(
        projector,
        missing_attributions=missing_attributions,
    )
    participant_fits, participant_predictions = fit_participants(
        trials,
        evaluator,
        candidates,
        candidates_by_property=candidates_by_property,
    )
    condition_fits = fit_conditions(
        trials,
        evaluator,
        candidates,
        candidates_by_property=candidates_by_property,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "screening": screening,
        "participant_fits": participant_fits,
        "condition_fits": condition_fits,
        "participant_predictions": participant_predictions,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    config = {
        "participant_dir": str(Path(participant_dir)),
        "assets_root": str(Path(assets_root)) if assets_root is not None else None,
        "missing_attributions": missing_attributions,
        "exp_properties": list(exp_properties) if exp_properties else None,
        "expected_test_qids": sorted(EXPECTED_TEST_QIDS),
        "n_candidates": len(candidates),
        "n_candidates_by_property": {
            exp_property: len(property_candidates)
            for exp_property, property_candidates in (
                candidates_by_property or {}
            ).items()
        },
        "candidate_space": {
            "aggregation": sorted({item.aggregation for item in candidates}),
            "normalize_by_i_max": sorted(
                {item.normalize_by_i_max for item in candidates}
            ),
            "confidence_scale": sorted(
                {item.confidence_scale for item in candidates}
            ),
            "confidence_intercept": sorted(
                {item.confidence_intercept for item in candidates}
            ),
            "comparison_C": sorted({item.comparison_C for item in candidates}),
            "max_features_attended": sorted(
                {item.max_features_attended for item in candidates}
            ),
            "guess_bias": sorted({item.guess_bias for item in candidates}),
            "lapse_rate": sorted({item.lapse_rate for item in candidates}),
            "use_exemplar_memory": sorted(
                {item.use_exemplar_memory for item in candidates}
            ),
            "memory_sensitivity": sorted(
                {item.memory_sensitivity for item in candidates}
            ),
            "memory_decay": sorted({item.memory_decay for item in candidates}),
            "retrieval_threshold": sorted(
                {
                    -np.inf if item.retrieval_threshold is None
                    else item.retrieval_threshold
                    for item in candidates
                }
            ),
        },
        "learned_parameters": list(LEARNED_PARAMETER_NAMES),
        "bic_formula": "n_parameters * log(n_trials) + 2 * total_nll",
        "n_eligible_participants": int(trials["participant_id"].nunique()),
        "n_screened_out": int((~screening["eligible"]).sum()),
        "n_human_test_trials": len(trials),
        "training_instance_ids": list(range(10)),
        "testing_instance_ids": list(range(10, 39)),
        "sequence": "fit 10 training feedback cases, then infer 29 testing cases",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    return outputs


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "server_runs" / "sim2real_attribution_sum"),
    )
    parser.add_argument("--assets-root", default=None)
    parser.add_argument(
        "--aggregations",
        default="attribution",
    )
    parser.add_argument(
        "--normalizations",
        default="raw",
        help="Comma-separated subset of raw,normalized",
    )
    parser.add_argument(
        "--confidence-scales",
        default="0.25,0.5,1,2,4,8",
    )
    parser.add_argument(
        "--confidence-intercepts",
        default="-2,-1,0,1,2",
    )
    parser.add_argument(
        "--comparison-cs",
        default="1000000",
        help=(
            "Inverse L2 strength for the delta->Higher logistic. Values at or "
            "below 100 shrink the slope to zero and make the model emit a flat "
            "0.5, because confidence_delta is on the order of 1e-4"
        ),
    )
    parser.add_argument(
        "--max-features-attended",
        default="4,12",
        help="Comma-separated raw-feature attention limits from 1 to 12",
    )
    parser.add_argument(
        "--guess-biases",
        default="0",
        help=(
            "Comma-separated guessing logits. Use --guess-biases=-1,0,1 "
            "when including negative values. sigmoid(0)=0.5."
        ),
    )
    parser.add_argument(
        "--lapse-rates",
        default="0",
        help="Comma-separated lapse mixture weights from 0 to 1",
    )
    parser.add_argument(
        "--memory-options",
        default="off,on",
        help="Comma-separated subset of off,on for CoAX exemplar memory",
    )
    parser.add_argument(
        "--memory-sensitivities",
        default="1,2,4",
        help="GCM sensitivities searched when exemplar memory is on",
    )
    parser.add_argument(
        "--memory-decays",
        default="0.5",
        help="Activation decay rates searched when exemplar memory is on",
    )
    parser.add_argument(
        "--retrieval-thresholds",
        default="none",
        help=(
            "Comma-separated retrieval cutoffs, or none for no cutoff"
        ),
    )
    parser.add_argument(
        "--exp-properties",
        default=None,
        help=(
            "Comma-separated conditions to fit, e.g. faithful. Omit to fit all. "
            "Use this to run one condition at a time with a grid sized for it"
        ),
    )
    parser.add_argument(
        "--condition-specific",
        action="store_true",
        help=(
            "Restrict guessing to sparse: faithful, robust and sparse_robust "
            "keep guess_bias=0 and lapse_rate=0"
        ),
    )
    parser.add_argument(
        "--missing-attributions",
        choices=("raise", "zero"),
        default="zero",
        help=(
            "zero is required to include truncated sparse_robust training "
            "instance 9; the assumption is recorded in run_config.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    normalizations = tuple(
        value == "normalized"
        for value in (item.strip() for item in args.normalizations.split(","))
        if value in {"raw", "normalized"}
    )
    if not normalizations:
        raise ValueError("--normalizations must include raw and/or normalized")
    grid_options = dict(
        aggregations=tuple(
            item.strip() for item in args.aggregations.split(",") if item.strip()
        ),
        normalize_options=normalizations,
        confidence_scales=_parse_float_list(args.confidence_scales),
        confidence_intercepts=_parse_float_list(args.confidence_intercepts),
        comparison_cs=_parse_float_list(args.comparison_cs),
        max_features_attended_options=tuple(
            int(item.strip())
            for item in args.max_features_attended.split(",")
            if item.strip()
        ),
        guess_biases=_parse_float_list(args.guess_biases),
        lapse_rates=_parse_float_list(args.lapse_rates),
        memory_options=tuple(
            item.strip() == "on"
            for item in args.memory_options.split(",")
            if item.strip() in {"on", "off"}
        ),
        memory_sensitivities=_parse_float_list(args.memory_sensitivities),
        memory_decays=_parse_float_list(args.memory_decays),
        retrieval_thresholds=tuple(
            None if item.strip().lower() == "none" else float(item.strip())
            for item in args.retrieval_thresholds.split(",")
            if item.strip()
        ),
    )
    if not grid_options["memory_options"]:
        raise ValueError("--memory-options must include off and/or on")
    candidates = candidate_grid(**grid_options)
    candidates_by_property = (
        condition_specific_candidate_grids(**grid_options)
        if args.condition_specific
        else None
    )
    outputs = run_fitting(
        participant_dir=args.participant_dir,
        output_dir=args.output_dir,
        assets_root=args.assets_root,
        candidates=candidates,
        candidates_by_property=candidates_by_property,
        exp_properties=(
            tuple(
                item.strip()
                for item in args.exp_properties.split(",")
                if item.strip()
            )
            if args.exp_properties
            else None
        ),
        missing_attributions=args.missing_attributions,
    )
    screening = outputs["screening"]
    print(
        f"eligible={int(screening['eligible'].sum())} "
        f"screened_out={int((~screening['eligible']).sum())} "
        f"candidates={len(candidates)} output={args.output_dir}"
    )
    print(outputs["condition_fits"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_TEST_QIDS",
    "EXP_PROPERTY_BY_EXPLANATION_TYPE",
    "AttributionSumCandidate",
    "CandidateInference",
    "load_screened_participant_trials",
    "attach_test_instances",
    "candidate_grid",
    "TrainedCandidateEvaluator",
    "fit_participants",
    "fit_conditions",
    "run_fitting",
]

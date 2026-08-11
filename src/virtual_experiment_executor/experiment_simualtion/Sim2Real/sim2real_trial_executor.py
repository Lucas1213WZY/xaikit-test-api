"""Drive the Sim2Real fitted attribution-sum model from a study's trials.

Mirrors ``CoAX/coax_trial_executor.py``: this module owns the data bridging and
the per-trial loop, while ``sim2real_study_runner`` reads the study object and
selects which trials to run.

Sim2Real is **counterfactual-only**. Each trial shows a participant one
instance plus an explanation, changes one feature, and asks whether the AI's
income prediction goes up. The model answers by comparing attribution sums
before and after the change (``confidence_delta = p_new - p_original``), so
there is no forward-simulation mode to fall back on -- unlike CoAX and CoXAM,
whose runners serve both tasks.

The condition IV is ``xai_property`` (faithful / sparse / robust /
sparse_robust), not ``xai_type``: Sim2Real's xai_type is ``attribution``
throughout, and the study varies which *property* the explanation was optimized
for. ``exp_property=None`` selects the unoptimized LIME baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.cognitive_models.cognitive_models.Sim2Real.gcm_strategies import (
    Sim2RealAttributionProjector,
    Sim2RealFittedAttributionSum,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

#: Explanation properties the corpus carries, matching
#: ``support_matrix.json``'s ``groups.xai_properties.sim2real``.
SIM2REAL_EXP_PROPERTIES: tuple[str, ...] = ("faithful", "sparse", "robust", "sparse_robust")

#: The unoptimized LIME baseline, which ``project(exp_property=None)`` selects.
BASELINE_EXP_PROPERTY = "baseline"

#: The model's own neutral defaults. **Not** what an unconfigured run should
#: use: at ``comparison_scale=1.0`` a typical ``confidence_delta`` of ~1e-4
#: maps to p≈0.5, so the model answers "increases" on every trial by tie-break
#: and the response carries no information. Kept only as the fallback when the
#: fitted table below cannot serve a condition.
NEUTRAL_SIM2REAL_PARAMS: dict[str, Any] = {
    "confidence_scale": 1.0,
    "confidence_intercept": 0.0,
    "comparison_scale": 1.0,
    "comparison_intercept": 0.0,
    "guess_bias": 0.0,
    "lapse_rate": 0.0,
    "use_exemplar_memory": False,
    "retrieval_threshold": None,
    "memory_sensitivity": 1.0,
    "memory_decay": 0.5,
    "aggregation": "attribution",
    "max_features_attended": 12,
}

#: Fitted-population parameter file behind :data:`FITTED_SIM2REAL_PARAMS`.
#: Produced by ``fit_sim2real_attribution_sum_to_participants.py``; git-ignored,
#: which is why the values below are baked into source rather than read at
#: runtime -- the same arrangement ``FITTED_COAX_PARAMS`` uses.
FITTED_PARAMETER_FILE = (
    REPO_ROOT / "server_runs" / "sim2real_attribution_sum" / "participant_fits.csv"
)

#: Per-condition parameters fitted to the Sim2Real study participants, so an
#: unconfigured run starts from the fitted population rather than from values
#: that make the model degenerate.
#:
#: ``comparison_scale`` uses the **median**: its distribution is heavily
#: right-skewed by participants whose fit selected ``comparison_C = 1e6``
#: (a hard threshold rather than a graded slope), which drags the mean far
#: above any individual. Other numerics use the mean; the two categoricals use
#: the modal choice.
#:
#: Fitted with ``normalize_by_i_max=False``, which is also this codebase's
#: default again: simulated forward accuracy measured against real human
#: accuracy on the same instances confirmed True (briefly the default) was
#: wrong -- it left three of four conditions unchanged but pushed
#: ``faithful`` from a close match to a much larger miss. Fits: 46
#: participants, 4320 candidates each, 10 training feedback cases then 29
#: test cases.
FITTED_SIM2REAL_PARAMS: dict[str, dict[str, Any]] = {
    # n = 12, mean model-participant test accuracy 0.716
    "faithful": {
        "confidence_scale": 5.1667,
        "confidence_intercept": -0.5833,
        "comparison_scale": 42.1271,
        "comparison_intercept": 0.1948,
        "aggregation": "attribution",
        "max_features_attended": 2,
    },
    # n = 12, mean model-participant test accuracy 0.463 -- the weakest fit of
    # the four; treat sparse-condition simulations with corresponding caution.
    "sparse": {
        "confidence_scale": 6.8333,
        "confidence_intercept": 1.6667,
        "comparison_scale": 36.0312,
        "comparison_intercept": -0.1223,
        "aggregation": "value_weighted",
        "max_features_attended": 2,
    },
    # n = 11, mean model-participant test accuracy 0.919
    "robust": {
        "confidence_scale": 7.3636,
        "confidence_intercept": 0.0,
        "comparison_scale": 91.0649,
        "comparison_intercept": 0.8522,
        "aggregation": "value_weighted",
        "max_features_attended": 2,
    },
    # n = 11, mean model-participant test accuracy 0.715
    "sparse_robust": {
        "confidence_scale": 5.8636,
        "confidence_intercept": -1.8182,
        "comparison_scale": 292.9607,
        "comparison_intercept": -0.3499,
        "aggregation": "value_weighted",
        "max_features_attended": 2,
    },
}

#: Backwards-compatible alias; the runner's real default is the fitted table.
DEFAULT_SIM2REAL_PARAMS = NEUTRAL_SIM2REAL_PARAMS

#: Carries study trial order through per-condition grouping.
ORDER_COLUMN = "__sim2real_trial_order"


def _canonical_exp_property(value: Any) -> Optional[str]:
    """A trial's ``xai_property`` level as the projector names it.

    Returns ``None`` for the unoptimized baseline, which is what
    ``Sim2RealAttributionProjector.project`` expects for that condition.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "baseline", "lime", "unoptimized", "unoptimised"}:
        return None
    if text in SIM2REAL_EXP_PROPERTIES:
        return text
    raise ValueError(
        f"Unknown Sim2Real explanation property {value!r}. "
        f"Use one of {SIM2REAL_EXP_PROPERTIES} or a baseline alias."
    )


def _trial_exp_property(trial: Mapping[str, Any]) -> Optional[str]:
    """Read the condition off a trial row, whichever column carries it."""
    for column in ("xai_property", "xai_properties", "explanation_property", "exp_property"):
        if column in trial and trial[column] is not None:
            return _canonical_exp_property(trial[column])
    return None


def build_sim2real_projector(
    *,
    assets_root: Optional[Path] = None,
    app_id: str = "adult_sim2real",
    model_name: str = "synthetic_ai",
    exp_method: str = "lime",
) -> Sim2RealAttributionProjector:
    """Load the Sim2Real corpus and explanation tables.

    Unlike CoAX and CoXAM, Sim2Real does not fit surrogates from a study's own
    prepared dataset: its stimuli, attributions and counterfactuals are a fixed
    published corpus, and the counterfactual ground truth lives in
    ``deltas.csv``. So the projector always reads the assets.
    """
    return Sim2RealAttributionProjector.from_assets(
        assets_root=assets_root,
        app_id=app_id,
        model_name=model_name,
        exp_method=exp_method,
    )


def fitted_sim2real_params(
    exp_property: Optional[str] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Constructor kwargs for one condition, from the fitted population.

    Falls back to :data:`NEUTRAL_SIM2REAL_PARAMS` for anything the fitted table
    does not carry, and entirely for the unoptimized baseline condition, which
    no participant was fitted on.
    """
    params = dict(NEUTRAL_SIM2REAL_PARAMS)
    params.update(FITTED_SIM2REAL_PARAMS.get(str(exp_property), {}))
    params.update(overrides)
    return params


#: Parameters the exemplar-similarity strategies take. They read feature values
#: rather than attributions, so none of the attribution-shaped fitted parameters
#: apply to them.
GCM_PARAM_NAMES = frozenset(
    {
        "k",
        "sensitivity",
        "always_attend_changed",
        "memory_decay",
        "retrieval_threshold",
        "guess_bias",
        "lapse_rate",
        "comparison_scale",
        "comparison_intercept",
        "comparison_C",
    }
)

#: Strategy names accepted by :func:`build_sim2real_model`.
SIM2REAL_STRATEGIES: tuple[str, ...] = (
    "attribution_sum",
    "sensitive_features",
    "salient_features",
)

#: Pick the strategy per condition instead of forcing one on all four.
AUTO_STRATEGY = "auto"

#: Which strategy each condition simulates with under :data:`AUTO_STRATEGY`.
#:
#: ``attribution_sum`` reads the explanation's attributions, so it needs the
#: changed feature to *have* a visible attribution. Measured on the corpus, that
#: holds for robust (100% of trials), faithful (76%) and sparse_robust (55%) --
#: but for ``sparse`` it holds on 6.9%. With nothing to read there its
#: confidence delta is a median of exactly 0.0, the response probability pins at
#: 0.4695 on every trial, and the model answers "no increase" every time: a
#: constant predictor scoring 0.310 against a real human 0.757.
#:
#: ``sparse`` therefore falls back to exemplar similarity over feature values,
#: which still has signal when the explanation does not. This mirrors what CoAX
#: already does -- its ``none`` condition is locked to SensitiveFeatures, and
#: ``attribution`` re-offers it whenever the explanation is hidden -- i.e. match
#: the strategy to the information actually available, and ``sparse`` meets that
#: test even though an explanation is technically on screen.
#:
#: The other three keep ``attribution_sum``: the exemplar strategies ignore the
#: explanation entirely, and ``xai_property`` *only* varies the explanation, so
#: using them everywhere would return one flat accuracy for all four conditions
#: and erase the manipulation the study exists to measure.
#: ``sparse_robust`` joins it for the same reason measured a different way: its
#: changed feature is visible on 55% of trials, and attribution_sum tops out at
#: 0.667 against a real human 0.920 -- and *falls* to 0.476 as comparison_scale
#: is lowered, so no setting of the attribution model reaches the human value.
#: Exemplar similarity scores 0.905 there.
#:
#: ``robust`` deliberately stays on attribution_sum despite simulating at 1.000
#: against a human 0.912. Its changed feature is visible on 100% of trials, so
#: the evidence is unambiguous and the model is simply right about all 21 of
#: them; the human 8.8% error rate is people slipping on easy trials, which a
#: deterministic model cannot express. Tuning comparison_scale does not help --
#: accuracy jumps 0.857 -> 1.000 between 10 and 12 with nothing in between,
#: because at the threshold three instances flip at once. Closing that gap
#: needs responses sampled from the response probability rather than
#: thresholded at 0.5 (which is also what would make ``lapse_rate`` do
#: anything), not a re-tuned scale.
SIM2REAL_STRATEGY_BY_PROPERTY: dict[str, str] = {
    "sparse": "sensitive_features",
    "sparse_robust": "sensitive_features",
}

#: Parameters for the exemplar strategies, per condition. Unlike
#: :data:`FITTED_SIM2REAL_PARAMS` these are *not* from the participant-level
#: fit, which only ever fitted attribution_sum: they come from a grid search
#: (sensitivity x k) scored against real human accuracy on the corpus's test
#: instances. sensitivity=30, k=3 puts sparse at 0.762 against a human 0.757,
#: and accuracy falls monotonically in sensitivity for k=3 and k=5 (0.905 ->
#: 0.810 -> 0.762 -> 0.714), so the match is a crossing of a real trend rather
#: than one lucky cell; k=1 is erratic and deliberately not used.
#:
#: Two honest limits. The grid was scored on the same ~21 test instances it is
#: measured against, so 0.762 is in-sample, not a held-out estimate. And those
#: instances make the resolution 1/21 ~ 0.048 -- the 0.005 gap to the human
#: value is a tenth of one trial, so this is "indistinguishable at the
#: achievable resolution", not an exact fit.
#: ``sparse_robust`` keeps CoAX's own default sensitivity of 10, which lands it
#: at 0.905 against a human 0.920 -- within a third of one trial at this
#: resolution, so there was nothing to tune away.
FITTED_SIM2REAL_GCM_PARAMS: dict[str, dict[str, Any]] = {
    "sparse": {"sensitivity": 30.0, "k": 3},
    "sparse_robust": {"sensitivity": 10.0, "k": 3},
}


def sim2real_strategy_for(exp_property: Any, requested: str = AUTO_STRATEGY) -> str:
    """The strategy one condition simulates with.

    An explicit ``requested`` other than ``auto`` wins for every condition, so a
    caller can still force one model across the study.
    """
    if requested and requested != AUTO_STRATEGY:
        return requested
    return SIM2REAL_STRATEGY_BY_PROPERTY.get(str(exp_property), "attribution_sum")


def sim2real_params_for(
    exp_property: Any,
    strategy: str,
    cognitive_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Condition parameters for ``strategy``, with caller overrides on top.

    The exemplar strategies take their own parameters and none of the fitted
    attribution ones, so they start from :data:`FITTED_SIM2REAL_GCM_PARAMS`
    rather than the attribution fit.
    """
    resolved: dict[str, Any] = {}
    if strategy != "attribution_sum":
        resolved.update(FITTED_SIM2REAL_GCM_PARAMS.get(str(exp_property), {}))
    resolved.update(cognitive_params or {})
    return resolved


def build_sim2real_model(
    params: Optional[Mapping[str, Any]] = None,
    *,
    exp_property: Optional[str] = None,
    strategy: str = "attribution_sum",
    projector: Optional[Sim2RealAttributionProjector] = None,
    normalize_by_i_max: bool = False,
):
    """Construct the model for one condition, fitted values topped up by ``params``.

    ``params`` wins over the fitted population, so a caller reproducing one
    participant passes that participant's own row and gets it exactly.

    ``strategy`` picks which cognitive model runs. It defaults to
    ``attribution_sum`` so existing simulations are unchanged, but the
    exemplar-similarity strategies are the ones that fit participants better
    wherever the changed feature has no visible attribution -- which is most of
    ``sparse`` and about half of ``sparse_robust``. They read feature values,
    so they take their own parameters and none of the fitted attribution ones.
    """
    if strategy not in SIM2REAL_STRATEGIES:
        raise ValueError(
            f"Unknown sim2real strategy {strategy!r}. Known: {list(SIM2REAL_STRATEGIES)}."
        )

    if strategy != "attribution_sum":
        from src.cognitive_models.cognitive_models.Sim2Real import (
            Sim2RealFittedSalientFeatures,
            Sim2RealFittedSensitiveFeatures,
        )

        cls = {
            "sensitive_features": Sim2RealFittedSensitiveFeatures,
            "salient_features": Sim2RealFittedSalientFeatures,
        }[strategy]
        kwargs = {
            key: value
            for key, value in (params or {}).items()
            if key in GCM_PARAM_NAMES
        }
        model = cls(**kwargs)
        # Unlike the attribution model, whose fitted parameters are baked into
        # FITTED_SIM2REAL_PARAMS, an exemplar strategy carries no parameters at
        # all until it has seen the training instances -- it *is* its memory.
        # Returning one unfitted yields an empty exemplar set, a confidence of
        # 0.5 on every instance, a zero delta and a constant 0.5 response.
        if projector is None:
            raise ValueError(
                f"strategy={strategy!r} must be fitted on the training instances "
                "before it can simulate, so build_sim2real_model needs projector=. "
                "Without it the model has no exemplars and answers 0.5 every trial."
            )
        training_pairs = projector.project_all(
            exp_property=exp_property,
            normalize_by_i_max=normalize_by_i_max,
            instance_ids=projector.instance_ids_for_split("training"),
        )
        model.fit(training_pairs)
        return model

    resolved = fitted_sim2real_params(exp_property)
    for key, value in (params or {}).items():
        if key in NEUTRAL_SIM2REAL_PARAMS:
            resolved[key] = value
    # missing_attributions="zero" keeps a trial whose source attribution is NaN
    # runnable; "raise" would abort a whole study for one gap in the corpus.
    return Sim2RealFittedAttributionSum(missing_attributions="zero", **resolved)


def sim2real_available_instance_ids(
    projector: Optional[Sim2RealAttributionProjector] = None,
    *,
    split: Optional[str] = None,
) -> list[int]:
    """Instance ids the corpus can serve, optionally for one raw split.

    Pass to ``study.generate_trials(allowed_instance_ids=...)`` so every trial
    references an instance the corpus actually carries -- the same preflight
    ``coxam_available_instance_ids`` exists for.
    """
    projector = projector or build_sim2real_projector()
    if split is None:
        return list(projector.instance_ids)
    return list(projector.instance_ids_for_split(split))


def _result_row(
    trial: Mapping[str, Any],
    comparison: Any,
    *,
    truth: Optional[int],
    dvs: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """One output row per trial, shaped like the CoAX/CoXAM runners'.

    The response is "does income increase?", so the DV is whether that matches
    the corpus's own feedback label. Fills any DV whose name mentions accuracy,
    the convention the other runners use.
    """
    probability = float(comparison.probability_income_increases)
    response = int(probability >= 0.5)
    correct = None if truth is None else int(response == int(truth))

    dv_columns = {
        name: correct
        for name in (dvs or {})
        if "accuracy" in name.lower() or "success" in name.lower()
    }
    return {
        **dv_columns,
        **trial,
        "exp_property": comparison.exp_property,
        "qid": comparison.qid,
        "original_attribution_sum": comparison.original_attribution_sum,
        "changed_attribution_sum": comparison.changed_attribution_sum,
        "p_original_high_income": comparison.p_original_high_income,
        "p_new_high_income": comparison.p_new_high_income,
        "confidence_delta": comparison.confidence_delta,
        "probability_income_increases": probability,
        "agent_response_increases": response,
        "ground_truth_increases": truth,
        "counterfactual_correct": correct,
        "changed_feature_visible": comparison.changed_feature_visible,
        "changed_feature_names": ", ".join(comparison.changed_feature_names),
        "retrieved_exemplar_count": comparison.retrieved_exemplar_count,
        "effective_lapse_rate": comparison.effective_lapse_rate,
    }


def run_sim2real_experiment_executor(
    trials: pd.DataFrame,
    *,
    projector: Optional[Sim2RealAttributionProjector] = None,
    model: Optional[Sim2RealFittedAttributionSum] = None,
    cognitive_params: Optional[Mapping[str, Any]] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    normalize_by_i_max: bool = False,
    strategy: str = AUTO_STRATEGY,
) -> pd.DataFrame:
    """Run each trial through the fitted model and return one row per trial.

    A **model per condition**: the fit selected different parameters for each
    explanation property (``robust`` uses a comparison scale of 91 where
    ``faithful`` uses 42, and three of the four selected ``value_weighted``
    aggregation where ``faithful`` did not), so one shared model would
    misrepresent three conditions out of four. Pass ``model`` to override that
    and use a single model everywhere.

    ``normalize_by_i_max`` defaults to False, matching how the baked
    ``FITTED_SIM2REAL_PARAMS`` were actually fitted. Measured against real
    human accuracy on the same instances, turning it on leaves three of the
    four conditions unchanged but pushes ``faithful`` from a close match to a
    much larger miss (0.677 vs a real 0.716 with it off; 0.861 with it on).
    """
    if trials.empty:
        return trials

    projector = projector or build_sim2real_projector()

    available = set(projector.instance_ids)
    requested = {int(value) for value in trials["instanceId"]}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"{len(missing)} trial instance(s) are outside the Sim2Real corpus "
            f"(e.g. {missing[:5]}). Constrain trial generation to it: "
            "study.generate_trials(allowed_instance_ids="
            "sim2real_available_instance_ids(), ...)."
        )

    models: dict[Optional[str], Sim2RealFittedAttributionSum] = {}
    rows: list[dict[str, Any]] = []
    for _, trial in trials.iterrows():
        record = trial.to_dict()
        instance_id = int(record["instanceId"])
        exp_property = _trial_exp_property(record)

        if model is not None:
            trial_model = model
        else:
            if exp_property not in models:
                # Resolved per condition, not once for the study: under
                # strategy="auto" sparse simulates with exemplar similarity and
                # the rest with attribution_sum (see
                # SIM2REAL_STRATEGY_BY_PROPERTY).
                trial_strategy = sim2real_strategy_for(exp_property, strategy)
                models[exp_property] = build_sim2real_model(
                    sim2real_params_for(exp_property, trial_strategy, cognitive_params),
                    exp_property=exp_property,
                    strategy=trial_strategy,
                    projector=projector,
                    normalize_by_i_max=normalize_by_i_max,
                )
            trial_model = models[exp_property]

        pair = projector.project(
            instance_id,
            exp_property=exp_property,
            normalize_by_i_max=normalize_by_i_max,
        )
        comparison = trial_model.compare(pair)
        truth = int(projector.increase_labels([instance_id])[0])
        rows.append(_result_row(record, comparison, truth=truth, dvs=dvs))
    return pd.DataFrame(rows)


__all__ = [
    "BASELINE_EXP_PROPERTY",
    "DEFAULT_SIM2REAL_PARAMS",
    "FITTED_PARAMETER_FILE",
    "FITTED_SIM2REAL_PARAMS",
    "NEUTRAL_SIM2REAL_PARAMS",
    "fitted_sim2real_params",
    "ORDER_COLUMN",
    "SIM2REAL_EXP_PROPERTIES",
    "build_sim2real_model",
    "build_sim2real_projector",
    "run_sim2real_experiment_executor",
    "sim2real_available_instance_ids",
]

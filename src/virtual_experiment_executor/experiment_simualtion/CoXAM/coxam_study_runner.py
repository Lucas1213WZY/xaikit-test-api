"""Run a xaikitTest study's generated trials through the CoXAM meta-policy.

Mirrors ``coax_study_runner.py``'s role for the CoAX integration. The shape
differs from CoAX in one structural way: a xaikitTest study trains exactly one
AI model on exactly one dataset, so there is exactly one CoXAM bundle for the
whole study (unlike CoAX, which can carry two XAI *methods* needing separate
explanation tables for the same dataset/model). What varies per participant is
the explanation *condition* (``xai_type``), which gates which of CoXAM's three
sub-strategies are legal for that participant's whole episode.

CoXAM's environment is episode-driven, not trial-driven (see
``coxam_trial_executor.TrialDrivenCombinedStrategyPolicyEnv``): one
participant's ordered trial sequence becomes one forced RL episode, so the
env's per-episode history/contribution-tracking state evolves the way it
would in a real run.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.ai_models.model_api import _labels_and_scores_from_predictions
from src.experiment_planner import select_trial_rows

from .coxam_trial_executor import (
    TrialDrivenCombinedStrategyPolicyEnv,
    _canonical_coxam_condition,
    _canonical_coxam_explanation_type,
    build_coxam_bundle,
    default_coxam_config,
    fit_coxam_surrogates,
    load_coxam_meta_policy,
    load_coxam_sub_policies,
    load_coxam_surrogates_from_assets,
)

#: Carries study trial order through the per-participant episodes so the
#: concatenated result reads in experiment order.
ORDER_COLUMN = "__coxam_trial_order"


def _trial_shows_explanation(trial: Mapping[str, Any]) -> bool:
    """Mirrors CoAX's own ``with_explanation = phase == 'training' or tested_w_xai``."""
    phase = str(trial.get("phase", "testing")).lower().strip()
    tested_w_xai = trial.get("tested_w_xai", False)
    if isinstance(tested_w_xai, str):
        tested_w_xai = tested_w_xai.strip().lower() in {"true", "1", "yes", "y"}
    return phase == "training" or bool(tested_w_xai)


def _trial_explanation_family(trial: Mapping[str, Any]) -> Any:
    """Which explanation family this one trial shows.

    ``shown_xai_type`` is preferred: the trial generator writes it per trial,
    resolving a multi-family condition (``xai_type='hybrid'``, which alternates
    between decision_tree and logistic_regression) down to the single family
    that trial actually displays. It equals ``xai_type`` for single-family
    conditions.

    Falls back to the condition itself for trial tables generated before
    ``shown_xai_type`` existed, and then to ``xai_method`` for designs authored
    before the xai_type steering reminder (support.py) -- CoXAM has one
    condition axis, not CoAX's two (xai_type "how" / xai_method "which"), so
    xai_type's own values already name the family.
    """
    return trial.get(
        "shown_xai_type",
        trial.get(
            "xai_type", trial.get("xai_method", trial.get("XAIType", trial.get("XAIMethod", "none")))
        ),
    )


#: Maps a design's canonical DV name to the task it names, so a call that
#: gets no explicit ``user_task`` and no per-trial ``user_task`` column (the
#: normal state for a design that names both DVs without also declaring
#: ``user_task`` as an explicit IV/CV -- see ``DesignExport.user_tasks``) can
#: still be routed correctly instead of silently defaulting to forward.
_TASK_BY_DV_NAME: dict[str, str] = {
    "forward_accuracy": "forward",
    "counterfactual_accuracy": "counterfactual",
}


def _tasks_implied_by_dvs(dvs: Optional[Mapping[str, Any]]) -> set[str]:
    if not dvs:
        return set()
    return {_TASK_BY_DV_NAME[name] for name in dvs if name in _TASK_BY_DV_NAME}


def _is_counterfactual(
    trials: pd.DataFrame,
    user_task: Optional[str],
    dvs: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Whether this run is counterfactual rather than forward simulation.

    An explicit ``user_task`` wins; otherwise the trials' own ``user_task``
    column decides, so a design that varies task as an IV routes correctly.

    Failing both of those, the DV names decide: a design that asks for only
    ``counterfactual_accuracy`` (no ``forward_accuracy``) means counterfactual
    simulation even with no ``user_task`` IV telling the trials apart, and
    defaulting that case to forward silently ran the wrong agent. A design
    naming *both* DVs (e.g. an ambiguous measure coerced to both, see
    ``COERCE_DVS_BY_FRAMEWORK``) still defaults to forward for a single
    ambiguous call -- CoXAM runs the two as separate agents, and a single
    call only ever produces one agent's numbers, so it cannot fill both DV
    columns correctly regardless of which one runs. ``_result_row`` is what
    actually guards against the original bug in this case: it fills only the
    DV(s) belonging to whichever task ran and warns about the rest, instead
    of filling every accuracy-shaped DV with that one task's numbers.
    """
    if user_task is not None:
        return "counterfactual" in str(user_task).strip().lower()
    if "user_task" in trials.columns:
        values = {str(value).strip().lower() for value in trials["user_task"].dropna().unique()}
        if values:
            if any("counterfactual" in value for value in values) and any(
                "forward" in value for value in values
            ):
                raise ValueError(
                    "Trials mix forward and counterfactual user_task values. CoXAM runs these "
                    "as separate agents, so select one task at a time via "
                    "run_coxam_study(user_task=...) or condition_filter."
                )
            return any("counterfactual" in value for value in values)
    implied = _tasks_implied_by_dvs(dvs)
    return implied == {"counterfactual"}


def _model_input_transform(dataset: Any, model_name: Optional[str] = None) -> Any:
    """The dataset's own raw -> model-input transform.

    This is what the CoXAM notebook's ``run_ai_prediction`` does::

        prepared = transform.prepare_instances_for_model(instance, one_hot_encode=True)
        preds = ai.predict(prepared)
        return np.argmax(preds[0])

    ``prepare_instances_for_model`` min-max scales continuous features against
    the dataset's *declared* ``feature_boundaries`` (not the observed range of
    ``X_raw``), leaves out-of-range values outside [0, 1] rather than clipping,
    and one-hot encodes categoricals. Re-deriving any of that by hand produces
    a different vector.

    The notebook hardcodes ``one_hot_encode=True`` because every dataset in its
    ``dataset_model_map`` is an ``mlp``. Here the model is a study choice, so
    the flag comes from ``requires_one_hot_encoding`` -- the same rule
    ``rebuild_model_arrays`` uses to build ``X_model`` in the first place --
    which returns ``True`` for mlp/mlp_tf and so agrees with the notebook.

    CoAX never needs any of this: it serves stored ``pred`` rows from a table.
    Only counterfactual simulation perturbs an instance to values no table
    holds, which is why it has to call the model live.
    """
    from src.ai_models import requires_one_hot_encoding

    source = dataset.split.dataset
    try:
        one_hot = requires_one_hot_encoding(str(model_name))
    except ValueError:
        # Unknown or unset model type: fall back to what X_model's own width says.
        expected_width = int(np.asarray(dataset.split.X_model).shape[1])
        probe = np.asarray(dataset.split.X_raw[:1], dtype=float)
        one_hot = next(
            (
                flag
                for flag in (True, False)
                if np.asarray(
                    source.prepare_instances_for_model(probe, one_hot_encode=flag)
                ).shape[1] == expected_width
            ),
            True,
        )

    def transform(raw_instance: Any) -> np.ndarray:
        raw = np.asarray(raw_instance, dtype=float).reshape(1, -1)
        return np.asarray(
            source.prepare_instances_for_model(raw, one_hot_encode=one_hot), dtype=float
        )

    return transform


def _label_from_predictions(predictions: Any) -> int:
    """One class label from a model's output.

    ``predict()`` in this repo returns *scores*, not labels -- the torch MLP
    returns the network's raw output and the Keras one returns probabilities
    (``predict_proba`` is a straight alias). Truncating those with ``int()``
    collapses every probability below 1.0 to class 0, so the conversion goes
    through the same helper ``classification_metrics`` uses.
    """
    labels, _scores = _labels_and_scores_from_predictions(predictions)
    return int(np.asarray(labels).reshape(-1)[0])


def _predict_label(
    trained_ai_model: Any, dataset: Any, raw_instance: Any, model_name: Optional[str] = None
) -> int:
    """The AI model's label for a single raw feature vector.

    Rebuilds the transform on every call, so use ``_model_input_transform``
    directly when predicting in a loop.
    """
    transform = _model_input_transform(dataset, model_name)
    return _label_from_predictions(trained_ai_model.predict(transform(raw_instance)))


def _trial_condition(trial: Mapping[str, Any]) -> str:
    """The episode-level condition gating which strategies are legal.

    Deliberately reads ``xai_type`` (the condition), never ``shown_xai_type``
    (the per-trial family): under ``hybrid`` the env must keep all three
    sub-strategies available for the whole episode, so collapsing the condition
    to whichever family a single trial happened to show would wrongly restrict
    it to one family.
    """
    xai_type = trial.get(
        "xai_type", trial.get("xai_method", trial.get("XAIType", trial.get("XAIMethod", "none")))
    )
    return _canonical_coxam_condition(xai_type)


def _participant_condition(rows: pd.DataFrame, group_key: Any) -> str:
    """The single CoXAM condition an episode's trials must share.

    ``group_key`` is whatever the episode was grouped by (participant, or
    participant+block) -- only used to name the offending group in the error.
    """
    conditions = {_trial_condition(row) for row in rows.to_dict("records")}
    if len(conditions) != 1:
        raise ValueError(
            f"Episode group {group_key!r} has trials spanning more than one "
            f"CoXAM condition ({sorted(conditions)}); CoXAM's environment gates "
            "legal strategies per episode, so one episode's trials must share a "
            "single condition. If conditions vary within a participant, group by "
            "block (a 'block' column on the trials) so each block becomes its "
            "own episode."
        )
    return next(iter(conditions))


def _result_row(
    trial: Mapping[str, Any],
    step: int,
    info: Mapping[str, Any],
    ai_prediction: Optional[int],
    dvs: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one output row.

    ``info`` (the env's own ``step()`` return) never echoes ``y_true`` -- it
    only carries the selected strategy's own guess (``selected_predicted_class``)
    and whether that guess matched (``selected_predicted_correct``). The AI's
    actual prediction is looked up by the caller directly from the bundle
    instead of being reconstructed from those two fields.
    """
    cognitive_correct_vs_ai = info.get("selected_predicted_correct")
    scorable = [name for name in (dvs or {}) if "accuracy" in name.lower()]
    # Forward simulation only ever produces agent-vs-AI agreement. A DV that
    # names a different task (counterfactual_accuracy) has to come from that
    # task's own run instead -- filling it here would be forward numbers
    # under another task's label, exactly the bug this guards against.
    mislabelled = [name for name in scorable if _TASK_BY_DV_NAME.get(name, "forward") != "forward"]
    if mislabelled and cognitive_correct_vs_ai is not None:
        warnings.warn(
            f"This is a forward-simulation run, but DV(s) {sorted(mislabelled)} name "
            "another task and will be left unfilled here. Run once per task (see "
            "DesignExport.user_tasks) and combine the results.",
            stacklevel=2,
        )
    dv_columns = {
        name: int(cognitive_correct_vs_ai)
        for name in scorable
        if name not in mislabelled and cognitive_correct_vs_ai is not None
    }
    return {
        **dv_columns,
        **trial,
        "phase": trial.get("phase"),
        "step": step,
        "condition_name": info.get("condition_name"),
        "explanation_type": info.get("explanation_type"),
        "selected_strategy": info.get("selected_strategy"),
        "ai_prediction": ai_prediction,
        "agent_prediction": info.get("selected_predicted_class"),
        "cognitive_correct_vs_ai": cognitive_correct_vs_ai,
        "prob_correct": info.get("selected_prob_correct"),
        "pred_time": info.get("selected_pred_time"),
    }


def run_coxam_episode(
    trials: pd.DataFrame,
    *,
    bundle: Any,
    condition_name: str,
    meta_policy: Any,
    uses_action_masks: bool,
    sub_policies: dict[str, Any],
    config: Any,
    policy_override: Optional[str] = None,
    fixed_eval_params: Optional[dict[str, float]] = None,
    dvs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Run one forced episode and return one DV row per trial.

    ``instances_per_episode`` is set to exactly ``len(trials)`` -- no padding
    or truncation. The env's ``progress`` observation scalar was trained at a
    fixed episode length (40); a shorter/longer forced episode changes its
    pace but not its meaning, an accepted, documented approximation.
    """
    if trials.empty:
        return trials

    env = TrialDrivenCombinedStrategyPolicyEnv(
        bundles=[bundle],
        config=dataclasses.replace(
            config,
            condition_name=condition_name,
            instances_per_episode=len(trials),
        ),
        sub_policies=sub_policies,
        training=False,
        fixed_eval_params=fixed_eval_params or {},
    )
    trial_records = trials.to_dict("records")
    env.queue_forced_episode(
        instance_ids=[int(row["instanceId"]) for row in trial_records],
        explanation_schedule=[
            _canonical_coxam_explanation_type(
                _trial_explanation_family(row),
                shown=_trial_shows_explanation(row),
            )
            for row in trial_records
        ],
    )
    obs, _ = env.reset()

    predictions_df = bundle.loader.AI_predictions_df
    ai_predictions_by_instance = {
        int(row["instanceId"]): (None if pd.isna(row["pred"]) else int(row["pred"]))
        for _, row in predictions_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for step, trial in enumerate(trial_records):
        if policy_override:
            action = env.strategy_slots.index(policy_override)
        elif uses_action_masks:
            action, _ = meta_policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        else:
            action, _ = meta_policy.predict(obs, deterministic=True)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        ai_prediction = ai_predictions_by_instance.get(int(trial["instanceId"]))
        rows.append(_result_row(trial, step, info, ai_prediction, dvs))
    return pd.DataFrame(rows)


def run_coxam_study(
    study: Any,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    trials: Optional[pd.DataFrame | Sequence[dict[str, Any]]] = None,
    data: Optional[Any] = None,
    source: str = "fit",
    user_task: Optional[str] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    policy_override: Optional[str] = None,
    eval_params: Optional[dict[str, float]] = None,
    store: bool = True,
) -> pd.DataFrame:
    """Execute a study's trials with CoXAM's meta-policy and return the step rows.

    Reads ``study.trials``/``study.data``/``study.trained_ai_model``/
    ``study.DVs`` off the study object, so it sits directly after
    ``study.train_AI_model(...)`` -- unlike ``run_coax_study``, it does not
    read ``study.combined_explanations``: that table carries the generic
    ``a{i}_i`` attribution schema CoAX also uses, produced by
    ``to_explanation_df()``, but CoXAM's own interpreter classes need the
    different ``coef_a{i}``/``tree_structure`` schema produced by
    ``to_explanation_table()``. So this function fits its own decision-tree
    and logistic-regression surrogates (see
    ``coxam_trial_executor.fit_coxam_surrogates``), mirroring
    ``tutorials/decision_tree_logistic_regression_experiment_workflow.ipynb``
    exactly, rather than reading anything the generic explanation stage
    produced.

    ``source`` picks where the surrogates come from:

    * ``"fit"`` (the default) fits fresh decision-tree and logistic-regression
      surrogates against this study's own trained model. Works for any dataset
      and any AI model.
    * ``"assets"`` loads the pre-generated tables from
      ``assets/ai_dataset/CoXAM`` and ``assets/explanations/CoXAM``. Faster and
      exactly reproduces the published corpus, but only covers the datasets the
      rules-vs-weights tutorial has been run against, each against an ``mlp``.
      Trials must reference instance ids that corpus actually carries.
    """
    trials_df = pd.DataFrame(study.trials if trials is None else trials).copy()
    if trials_df.empty:
        raise RuntimeError("No generated trials to run. Call study.generate_trials(...) first.")

    dataset = data if data is not None else study.data
    if dataset is None:
        raise RuntimeError("No prepared dataset available. Call study.prepare_dataset(...) first.")

    if source not in {"fit", "assets"}:
        raise ValueError(f"source must be 'fit' or 'assets', not {source!r}.")

    trained_ai_model = getattr(study, "trained_ai_model", None)
    if source == "fit" and trained_ai_model is None:
        raise RuntimeError("No trained AI model available. Call study.train_AI_model(...) first.")

    resolved_dvs = study.DVs if dvs is None else dvs

    selected = select_trial_rows(
        trials_df, mode, participant_id=participant_id, condition_filter=condition_filter,
    ).copy()
    if selected.empty:
        return selected
    selected[ORDER_COLUMN] = range(len(selected))

    app_id = str(dataset.dataset_id)
    model_name = str(getattr(study, "model_name", None) or "model")
    raw_feature_names = list(dataset.raw_feature_names)

    # The surrogates must cover every instance a generated trial could
    # reference, not just a capped sample: generate_trials() can draw from the
    # whole prepared dataset, and a trial whose instance was never "explained"
    # would fail bundle lookup at episode time. Fitting DT/LR surrogates is
    # cheap even over the full dataset (unlike SHAP/LIME's per-instance kernel
    # methods), so there is no accuracy/cost reason to sample here.
    #
    # trained_ai_model.predict(...) needs the model-ready matrix (normalized/
    # one-hot) it was trained on, but the surrogates must be fit against raw
    # feature values: CoXAM's env always evaluates apply_to_instance() with
    # normalize=False (true raw values), so a surrogate fit against the
    # normalized matrix would compare raw inputs against thresholds/bounds
    # calibrated for the [0, 1] range instead -- confirmed empirically that
    # X_test and X_test_raw differ (e.g. 0.212 vs 9.9 for the same instance).
    if source == "assets":
        surrogates = load_coxam_surrogates_from_assets(
            app_id=app_id, study_feature_names=raw_feature_names,
        )
        features = surrogates["features"]
    else:
        surrogates = fit_coxam_surrogates(
            app_id=app_id,
            model_name=model_name,
            X=dataset.split.X_raw,
            instance_ids=np.asarray(dataset.split.raw_instance_ids),
            trained_ai_model=trained_ai_model,
            ai_model_input=dataset.split.X_model,
            feature_names=raw_feature_names,
        )
        features = pd.DataFrame(
            {"dataId": app_id, "instanceId": dataset.split.raw_instance_ids}
        )
        for index in range(len(raw_feature_names)):
            features[f"x{index}"] = dataset.split.X_raw[:, index].astype(float)

    bundle = build_coxam_bundle(
        app_id=app_id,
        model_name=model_name,
        features=features,
        predictions=surrogates["predictions"],
        lr_explanations=surrogates["lr_explanations"],
        dt_explanations=surrogates["dt_explanations"],
        metadata=surrogates["metadata"],
    )

    # The asset corpus serves a narrower instance range than a study's own
    # split (e.g. wine_quality assets carry ids 0-399 against a prepared
    # dataset spanning 0-1598), so a trial can reference an instance the
    # corpus has never seen. Caught here rather than deep inside the episode,
    # where the failure would name neither the cause nor the fix.
    available = set(bundle.instance_ids)
    requested = {int(value) for value in selected["instanceId"]}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"{len(missing)} trial instance(s) are outside the CoXAM "
            f"{'asset corpus' if source == 'assets' else 'bundle'} for "
            f"appId={app_id!r} (e.g. {missing[:5]}). "
            + (
                "Either pass source='fit' to fit surrogates over this study's own "
                "dataset, or constrain trial generation to the corpus: "
                "study.generate_trials(allowed_instance_ids="
                f"coxam_available_instance_ids({app_id!r}), ...)."
                if source == "assets"
                else "The bundle and the trial table disagree about instance ids."
            )
        )

    if _is_counterfactual(selected, user_task, resolved_dvs):
        # Counterfactual simulation is a different trained agent with a
        # different observation space and a different DV, so it has its own
        # runner. Bundle construction and the coverage preflight above are
        # shared, which is why the dispatch happens here rather than earlier.
        from .coxam_counterfactual_runner import run_coxam_counterfactual_study

        # Resolved once: the transform refits its one-hot encoder per call, and
        # an episode predicts twice per trial.
        to_model_input = _model_input_transform(dataset, model_name)

        def predict_fn(raw_instance):
            return _label_from_predictions(trained_ai_model.predict(to_model_input(raw_instance)))

        return run_coxam_counterfactual_study(
            study,
            bundle=bundle,
            predict_fn=predict_fn,
            trials=selected.drop(columns=[ORDER_COLUMN]),
            dvs=resolved_dvs,
            cognitive_params=eval_params,
            store=store,
        )

    config = default_coxam_config()
    meta_policy, uses_action_masks = load_coxam_meta_policy()
    sub_policies = load_coxam_sub_policies(config)

    # One episode per (participant, block): xai_method/xai_type is commonly a
    # within-subject block IV (see the design tutorial), so one participant's
    # trials can legitimately span more than one CoXAM condition across
    # separate blocks. Grouping by block respects the real experimental
    # boundary even if the same condition happens to recur in two
    # non-adjacent blocks for the same participant, rather than merging them
    # into one artificially long episode.
    group_columns = [
        column for column in ("participantId", "block") if column in selected.columns
    ]
    episode_groups = (
        selected.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [(None, selected)]
    )

    runs = []
    for group_key, group_rows in episode_groups:
        ordered = group_rows.sort_values(ORDER_COLUMN, kind="stable")
        condition_name = _participant_condition(ordered, group_key)
        runs.append(
            run_coxam_episode(
                ordered,
                bundle=bundle,
                condition_name=condition_name,
                meta_policy=meta_policy,
                uses_action_masks=uses_action_masks,
                sub_policies=sub_policies,
                config=config,
                policy_override=policy_override,
                fixed_eval_params=eval_params,
                dvs=resolved_dvs,
            )
        )

    results = pd.concat(runs, ignore_index=True, sort=False)
    if ORDER_COLUMN in results.columns:
        results = (
            results.sort_values(ORDER_COLUMN, kind="stable")
            .drop(columns=[ORDER_COLUMN])
            .reset_index(drop=True)
        )
    unscored = int(results["cognitive_correct_vs_ai"].isna().sum()) if "cognitive_correct_vs_ai" in results else 0
    if unscored:
        warnings.warn(
            f"{unscored} of {len(results)} recorded steps could not be scored "
            "against the AI. Those rows carry no DV value.",
            stacklevel=2,
        )
    if store:
        study.simulated_results = results
    return results


__all__ = [
    "ORDER_COLUMN",
    "run_coxam_episode",
    "run_coxam_study",
]

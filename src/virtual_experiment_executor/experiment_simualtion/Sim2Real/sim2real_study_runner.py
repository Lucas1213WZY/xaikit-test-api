"""Execute a xaikitTest study's trials with the Sim2Real fitted model.

The entry point ``server/pipeline.py`` and ``xaikitTest.run_experiment`` call,
mirroring ``run_coax_study`` / ``run_coxam_study``: everything the run needs is
read off the study object, and each argument can be overridden to run a subset
without touching stored study state.

Sim2Real is counterfactual-only -- see ``sim2real_trial_executor`` for why --
so unlike the other two runners there is no ``user_task`` switch.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.experiment_planner import select_trial_rows

from .sim2real_trial_executor import (
    AUTO_STRATEGY,
    ORDER_COLUMN,
    build_sim2real_projector,
    run_sim2real_experiment_executor,
)


def run_sim2real_study(
    study: Any,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    trials: Optional[pd.DataFrame | Sequence[dict[str, Any]]] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    cognitive_params: Optional[Mapping[str, Any]] = None,
    normalize_by_i_max: bool = False,
    strategy: str = AUTO_STRATEGY,
    store: bool = True,
) -> pd.DataFrame:
    """Execute a study's trials with Sim2Real participants.

    Reads ``study.trials`` and ``study.DVs`` off the study object, so the call
    sits directly after ``study.generate_trials(...)``. Unlike the CoAX and
    CoXAM runners it needs neither ``study.data`` nor a trained AI model: the
    Sim2Real stimuli, attributions, counterfactuals and ground-truth feedback
    labels are a fixed published corpus, loaded by the projector.

    ``mode`` takes the same values as ``study.run_experiment``:
    ``trial_by_trial``, ``participant_by_participant``, ``whole_condition``
    (with ``condition_filter``) or ``whole_experiment``.

    ``strategy`` picks the cognitive model: ``attribution_sum`` (the default,
    and the best fit wherever the changed feature is visible),
    ``sensitive_features`` or ``salient_features`` (which read feature values
    and fit better where it is not).

    ``cognitive_params`` overrides the model's defaults; anything omitted keeps
    the default. Pass the values the participant-level fitter selected to
    reproduce a fitted participant.

    With ``store=True`` the result is written back to
    ``study.simulated_results``, which ``save_results``, ``analyze_iv_dv`` and
    the plotting helpers read.
    """
    trials_df = pd.DataFrame(study.trials if trials is None else trials).copy()
    if trials_df.empty:
        raise RuntimeError("No generated trials to run. Call study.generate_trials(...) first.")

    selected = select_trial_rows(
        trials_df, mode, participant_id=participant_id, condition_filter=condition_filter,
    ).copy()
    if selected.empty:
        return selected
    selected[ORDER_COLUMN] = range(len(selected))

    # The corpus is loaded once and shared; the executor builds one model per
    # explanation property, because the fit selected different parameters for
    # each condition.
    projector = build_sim2real_projector()

    results = run_sim2real_experiment_executor(
        selected,
        projector=projector,
        cognitive_params=cognitive_params,
        dvs=study.DVs if dvs is None else dvs,
        normalize_by_i_max=normalize_by_i_max,
        strategy=strategy,
    )

    if ORDER_COLUMN in results.columns:
        results = (
            results.sort_values(ORDER_COLUMN, kind="stable")
            .drop(columns=[ORDER_COLUMN])
            .reset_index(drop=True)
        )
    if store:
        study.simulated_results = results
    return results


__all__ = ["run_sim2real_study"]

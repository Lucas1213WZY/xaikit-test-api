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
from src.virtual_experiment_executor.participant_pools import (
    draw_participant_parameters,
    draws_to_frame,
    is_diverse_mode,
)

from .sim2real_trial_executor import (
    AUTO_STRATEGY,
    ORDER_COLUMN,
    _trial_exp_property,
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
    sampling_seed: int = 0,
    sampling_replace: Optional[bool] = None,
    parameter_pool: Optional[pd.DataFrame] = None,
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
    (with ``condition_filter``), ``whole_experiment`` or
    ``diverse_participant``.

    ``diverse_participant`` runs the same trials as ``whole_experiment`` but
    deals every participant its own parameters, drawn from the 46 fitted study
    participants in ``assets/human_data/Sim2Real/sim2real_participant_fits.csv``
    and filtered to that participant's own ``xai_property``. Without it the
    corpus serves every participant the same test instances and the model is
    deterministic, so a condition's 12 participants return 12 identical
    accuracies -- a within-condition SD of exactly 0.0, which makes every
    between-condition t statistic meaningless. ``sampling_seed`` makes the
    assignment reproducible, ``sampling_replace=True`` draws i.i.d. instead of
    dealing distinct people, and ``parameter_pool`` substitutes your own fitted
    table.

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

    participant_draws = (
        _draw_sim2real_participants(
            selected,
            seed=sampling_seed,
            replace=sampling_replace,
            pool=parameter_pool,
        )
        if is_diverse_mode(mode)
        else None
    )

    results = run_sim2real_experiment_executor(
        selected,
        projector=projector,
        cognitive_params=cognitive_params,
        dvs=study.DVs if dvs is None else dvs,
        normalize_by_i_max=normalize_by_i_max,
        strategy=strategy,
        participant_draws=participant_draws,
    )

    if ORDER_COLUMN in results.columns:
        results = (
            results.sort_values(ORDER_COLUMN, kind="stable")
            .drop(columns=[ORDER_COLUMN])
            .reset_index(drop=True)
        )
    if store:
        study.simulated_results = results
        if participant_draws is not None:
            study.participant_parameters = draws_to_frame(participant_draws)
    return results


def _draw_sim2real_participants(
    trials: pd.DataFrame,
    *,
    seed: int,
    replace: Optional[bool],
    pool: Optional[pd.DataFrame],
) -> dict[Any, Any]:
    """One fitted human per virtual participant, per explanation property.

    Drawn per condition cell rather than once for the study, so a participant
    only ever receives parameters fitted to somebody who was actually in that
    condition. ``xai_property`` is between-subjects in the published design, so
    in practice this is one draw per participant; a within-subject variant would
    give the same participant one row per property it sees, which is the correct
    reading of "fitted to this condition" either way.

    The seed is offset by the condition so two conditions do not deal the same
    pool rows in the same order.
    """
    draws: dict[Any, Any] = {}
    if "participantId" not in trials.columns:
        return draws
    conditions = trials.assign(
        __exp_property=[_trial_exp_property(row) for row in trials.to_dict("records")]
    )
    for offset, (exp_property, rows) in enumerate(
        conditions.groupby("__exp_property", sort=True, dropna=False)
    ):
        participants = sorted(rows["participantId"].dropna().unique().tolist(), key=str)
        draws.update(
            draw_participant_parameters(
                "sim2real",
                condition={"exp_property": exp_property},
                participants=participants,
                seed=int(seed) + offset,
                replace=replace,
                pool=pool,
            )
        )
    return draws


__all__ = ["run_sim2real_study"]

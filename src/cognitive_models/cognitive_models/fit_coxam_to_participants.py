"""Fit CoXAM's forward meta-policy to each study participant by NLL, then BIC.

Mirrors ``fit_sim2real_attribution_sum_to_participants``: negative log-likelihood
of the *participant's own response* selects the parameters, and BIC -- computed
from the winning fit's total NLL -- compares models. Outputs are shaped like
CoAX's published tables so the three studies can be plotted side by side.

**What is fitted.** Only the three parameters CoXAM leaves free at evaluation
time (``decision_noise``, ``memory_recall_threshold``, ``opportunity_cost``, see
``placeholder.COXAM_FORWARD_PARAMS``) plus a ``lapse_rate`` that belongs to the
*link* between CoXAM's output and a human response, not to CoXAM itself. No
CoXAM model, policy or checkpoint is modified or retrained.

**Forward only.** CoXAM's meta-policy is forward-simulation only; the
counterfactual strategies are a separate agent that bypasses it. Fitting those
needs ``coxam_counterfactual_runner`` and is deliberately out of scope here.

**Why a lapse rate is needed.** A strategy can return a degenerate ``[1, 0]``
distribution, which makes the NLL of a disagreeing human response infinite. The
lapse mixes the model toward chance, which is the standard observation-model fix
and keeps every candidate comparable. ``0.0`` is deliberately not searched: it
leaves those trials at infinity, so it could only ever lose.

**The rollout is deterministic** for fixed parameters, verified by running the
same episode twice, so the NLL is exact rather than a Monte-Carlo estimate. The
lapse never changes the rollout, only the probabilities it produces, so each of
the model candidates is rolled out once and every lapse is then swept for free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HUMAN_TRIALS_FILE = REPO_ROOT / "assets" / "human_data" / "CoXAM" / "coxam_human_trials.csv"

#: The study's condition labels -> CoXAM's own condition vocabulary.
CONDITION_ALIASES = {
    "DT": "decision_tree",
    "LR": "linear_regression",
    "Hybrid": "hybrid",
}

#: The study's per-trial explanation labels -> CoXAM's explanation *families*.
#: Note this vocabulary is not the condition one above: CoXAM calls the LR
#: condition ``linear_regression`` but the LR explanation family
#: ``logistic_regression``, and each rejects the other's spelling.
XAI_TYPE_ALIASES = {"DT": "decision_tree", "LR": "logistic_regression"}

#: Ranges come from the run config of the checkpoint actually loaded, not from
#: ``CombinedPolicyConfig``'s dataclass defaults -- ``memory_recall_threshold``
#: is [-1.0, 2.0] in every run config and [-5.0, 2.0] in the dataclass.
DECISION_NOISE_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7)
MEMORY_RECALL_THRESHOLD_VALUES = (-1.0, -0.25, 0.5, 1.25, 2.0)
OPPORTUNITY_COST_VALUES = (0.0, 0.01, 0.02)

#: Swept analytically once a candidate has been rolled out, so it costs nothing.
LAPSE_RATE_VALUES = (0.01, 0.025, 0.05, 0.1, 0.2)

#: Only ``decision_noise`` and ``opportunity_cost`` reach the strategy through
#: the env's parameter vector; ``memory_recall_threshold`` builds the memory.
MODEL_PARAMETER_NAMES = ("decision_noise", "memory_recall_threshold", "opportunity_cost")

#: Hard ceiling on worker processes. These rollouts are pure CPU and hold every
#: worker at ~100%; four of them overheated this laptop, so the cap is two and
#: a larger ``--workers`` is clamped rather than honoured. Prefer a longer run
#: at lower intensity over a shorter one that cooks the machine.
MAX_WORKERS = 2


@dataclass(frozen=True)
class CoxamCandidate:
    """One point in the searched parameter space."""

    decision_noise: float
    memory_recall_threshold: float
    opportunity_cost: float

    def eval_params(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in MODEL_PARAMETER_NAMES}


def candidate_grid(
    *,
    decision_noises: Sequence[float] = DECISION_NOISE_VALUES,
    memory_recall_thresholds: Sequence[float] = MEMORY_RECALL_THRESHOLD_VALUES,
    opportunity_costs: Sequence[float] = OPPORTUNITY_COST_VALUES,
) -> list[CoxamCandidate]:
    """Every combination of the three evaluation-time parameters."""
    return [
        CoxamCandidate(noise, threshold, cost)
        for noise in decision_noises
        for threshold in memory_recall_thresholds
        for cost in opportunity_costs
    ]


def count_free_parameters(
    candidates: Sequence[CoxamCandidate],
    lapse_rates: Sequence[float],
) -> int:
    """Charge BIC for the fields that actually vary in the search.

    A pinned field is not a fitted degree of freedom, so pinning one makes it
    free -- the same rule the sim2real fitter uses.
    """
    varying = sum(
        1
        for name in MODEL_PARAMETER_NAMES
        if len({getattr(candidate, name) for candidate in candidates}) > 1
    )
    return varying + (1 if len(set(lapse_rates)) > 1 else 0)


# -- human trials ---------------------------------------------------------


def load_coxam_human_forward_trials(
    *,
    path: Path = HUMAN_TRIALS_FILE,
    drop_excluded: bool = True,
) -> pd.DataFrame:
    """The forward-phase human trials, one row per trial.

    ``Response`` is a signed confidence on [-100, 100]; its *sign* is the class
    the participant chose, which is what ``Response==AI`` in the study's own
    table binarises. It is never 0 in this dataset, so the sign is never
    ambiguous, and that is asserted rather than assumed.
    """
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["Phase"] == "forward"].copy()
    if drop_excluded and "Exclude" in frame.columns:
        frame = frame[frame["Exclude"] != 1]

    frame = frame[frame["Response"].notna() & frame["Participant Id"].notna()]
    if (frame["Response"] == 0).any():
        raise ValueError("Response == 0 has no sign, so the chosen class is ambiguous.")

    frame["participant_id"] = frame["Participant Id"].astype(int)
    frame["human_label"] = (frame["Response"] > 0).astype(int)
    frame["ai_label"] = frame["AI prediction"].astype(int)
    frame["condition"] = frame["Condition"].map(CONDITION_ALIASES)
    if frame["condition"].isna().any():
        unknown = sorted(frame.loc[frame["condition"].isna(), "Condition"].unique())
        raise ValueError(f"Unmapped CoXAM condition(s): {unknown}")
    frame["shown_xai_type"] = frame["XAIType"].map(XAI_TYPE_ALIASES)
    frame["tested_w_xai"] = frame["Tested w/ XAI"].eq("w/ XAI")
    return frame.reset_index(drop=True)


# -- one participant ------------------------------------------------------


def _model_probability_class_one(episode: pd.DataFrame) -> np.ndarray:
    """P(CoXAM answers class 1) per trial.

    The env reports ``prob_correct`` = P(the class it was scored against), and
    that class is the AI's prediction -- the thing a participant is asked to
    predict. For a binary task the full distribution follows, so no Monte-Carlo
    resampling of the policy is needed.
    """
    prob_correct = episode["prob_correct"].to_numpy(dtype=float)
    ai_label = episode["ai_label"].to_numpy(dtype=float)
    probability = np.where(ai_label == 1, prob_correct, 1.0 - prob_correct)
    # A strategy that produced no distribution reports -1 and is treated as
    # chance rather than silently scored as a confident wrong answer.
    return np.where(np.isnan(probability), 0.5, probability)


def negative_log_likelihood(
    probability_class_one: np.ndarray,
    human_label: np.ndarray,
    lapse_rate: float,
) -> float:
    """Mean Bernoulli NLL of the participant's own responses."""
    mixed = (1.0 - lapse_rate) * probability_class_one + lapse_rate * 0.5
    likelihood = np.where(human_label == 1, mixed, 1.0 - mixed)
    return float(-np.mean(np.log(likelihood)))


def _episode_for_candidate(
    trials: pd.DataFrame,
    candidate: CoxamCandidate,
    *,
    runtime: Mapping[str, Any],
    condition: str,
) -> pd.DataFrame:
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
        run_coxam_episode,
    )

    episode_trials = pd.DataFrame(
        {
            "participantId": trials["participant_id"].to_numpy(),
            "instanceId": trials["Instance Id"].astype(int).to_numpy(),
            "phase": "testing",
            "tested_w_xai": trials["tested_w_xai"].to_numpy(),
            "shown_xai_type": trials["shown_xai_type"].to_numpy(),
        }
    )
    episode = run_coxam_episode(
        episode_trials,
        bundle=runtime["bundles"][trials["dataId"].iloc[0]],
        condition_name=condition,
        meta_policy=runtime["meta_policy"],
        uses_action_masks=runtime["uses_action_masks"],
        sub_policies=runtime["sub_policies"],
        config=runtime["config"],
        fixed_eval_params=candidate.eval_params(),
    )
    episode = episode.reset_index(drop=True)
    episode["ai_label"] = trials["ai_label"].to_numpy()
    return episode


def fit_one_participant(
    trials: pd.DataFrame,
    *,
    runtime: Mapping[str, Any],
    candidates: Sequence[CoxamCandidate],
    lapse_rates: Sequence[float] = LAPSE_RATE_VALUES,
) -> dict[str, Any]:
    """Search the grid for one participant on one dataset, keeping the best NLL.

    Every candidate is rolled out once; the lapse rates are then swept over the
    resulting probabilities, which costs nothing because the lapse cannot change
    the rollout.
    """
    trials = trials.sort_values("Trial Index").reset_index(drop=True)
    human_label = trials["human_label"].to_numpy()
    condition = trials["condition"].iloc[0]

    best: Optional[dict[str, Any]] = None
    for candidate in candidates:
        episode = _episode_for_candidate(trials, candidate, runtime=runtime, condition=condition)
        probability = _model_probability_class_one(episode)
        for lapse_rate in lapse_rates:
            nll = negative_log_likelihood(probability, human_label, lapse_rate)
            if best is None or nll < best["nll"]:
                best = {
                    "nll": nll,
                    "candidate": candidate,
                    "lapse_rate": lapse_rate,
                    "probability": probability,
                    "episode": episode,
                }

    assert best is not None, "candidate grid was empty"
    probability = best["probability"]
    mixed = (1.0 - best["lapse_rate"]) * probability + best["lapse_rate"] * 0.5
    model_label = (mixed > 0.5).astype(int)
    n_trials = len(trials)
    n_parameters = count_free_parameters(candidates, lapse_rates)

    fit = {
        "participant_id": int(trials["participant_id"].iloc[0]),
        "dataId": trials["dataId"].iloc[0],
        "condition": trials["Condition"].iloc[0],
        "coxam_condition": condition,
        "complexity": trials["Complexity"].iloc[0],
        "n_trials": n_trials,
        **asdict(best["candidate"]),
        "lapse_rate": best["lapse_rate"],
        "nll": best["nll"],
        "total_nll": best["nll"] * n_trials,
        "n_parameters": n_parameters,
        "bic": n_parameters * np.log(n_trials) + 2.0 * best["nll"] * n_trials,
        "accuracy_model_vs_participant": float(np.mean(model_label == human_label)),
        "accuracy_participant_vs_ai": float(np.mean(human_label == trials["ai_label"].to_numpy())),
        "accuracy_model_vs_ai": float(np.mean(model_label == trials["ai_label"].to_numpy())),
        "brier_model_vs_participant": float(np.mean((mixed - human_label) ** 2)),
    }

    predictions = pd.DataFrame(
        {
            "participant_id": fit["participant_id"],
            "dataId": fit["dataId"],
            "condition": fit["condition"],
            "trial_index": trials["Trial Index"].to_numpy(),
            "instanceId": trials["Instance Id"].astype(int).to_numpy(),
            "human_response": trials["Response"].to_numpy(),
            "human_label": human_label,
            "ai_label": trials["ai_label"].to_numpy(),
            "dt_label": trials["DT prediction"].to_numpy(),
            "lr_label": trials["LR prediction"].to_numpy(),
            "selected_strategy": best["episode"]["selected_strategy"].to_numpy(),
            "model_probability_class_one": mixed,
            "model_label": model_label,
        }
    )
    return {"fit": fit, "predictions": predictions}


# -- runtime --------------------------------------------------------------

_RUNTIME: dict[str, Any] = {}


def build_runtime(data_ids: Iterable[str]) -> dict[str, Any]:
    """Load the policies once and one CoXAM bundle per dataset."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        build_coxam_bundle,
        default_coxam_config,
        load_coxam_meta_policy,
        load_coxam_sub_policies,
        load_coxam_surrogates_from_assets,
    )

    config = default_coxam_config()
    meta_policy, uses_action_masks = load_coxam_meta_policy()
    bundles = {}
    for data_id in sorted(set(data_ids)):
        surrogates = load_coxam_surrogates_from_assets(app_id=data_id)
        bundles[data_id] = build_coxam_bundle(
            app_id=data_id,
            model_name="mlp",
            features=surrogates["features"],
            predictions=surrogates["predictions"],
            lr_explanations=surrogates["lr_explanations"],
            dt_explanations=surrogates["dt_explanations"],
            metadata=surrogates["metadata"],
        )
    return {
        "config": config,
        "meta_policy": meta_policy,
        "uses_action_masks": uses_action_masks,
        "sub_policies": load_coxam_sub_policies(config),
        "bundles": bundles,
    }


def _worker_init(data_ids: list[str], candidates: list[CoxamCandidate], lapse_rates: list[float]) -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    _RUNTIME["runtime"] = build_runtime(data_ids)
    _RUNTIME["candidates"] = candidates
    _RUNTIME["lapse_rates"] = lapse_rates


def _worker_fit(payload: tuple[Any, Any, bytes]) -> dict[str, Any]:
    _participant, _data_id, trials_json = payload
    trials = pd.read_json(StringIO(trials_json), orient="split")
    return fit_one_participant(
        trials,
        runtime=_RUNTIME["runtime"],
        candidates=_RUNTIME["candidates"],
        lapse_rates=_RUNTIME["lapse_rates"],
    )


# -- the whole study ------------------------------------------------------


def fit_participants(
    trials: Optional[pd.DataFrame] = None,
    *,
    candidates: Optional[Sequence[CoxamCandidate]] = None,
    lapse_rates: Sequence[float] = LAPSE_RATE_VALUES,
    n_workers: Optional[int] = None,
    limit: Optional[int] = None,
    progress: bool = True,
    checkpoint_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Fit every participant-dataset group and collect the results.

    ``n_workers`` is clamped to :data:`MAX_WORKERS`; see its note on heat.

    Each group's result is written to ``checkpoint_dir`` as it completes and
    re-read on a later run, so stopping the job -- which is the expected way to
    respond to an overheating laptop -- costs only the group in flight rather
    than the whole sweep.
    """
    from concurrent.futures import ProcessPoolExecutor

    trials = load_coxam_human_forward_trials() if trials is None else trials
    candidates = list(candidates if candidates is not None else candidate_grid())
    lapse_rates = list(lapse_rates)
    requested = n_workers or MAX_WORKERS
    n_workers = max(1, min(requested, MAX_WORKERS))
    if requested > MAX_WORKERS:
        print(f"  workers clamped {requested} -> {n_workers} (MAX_WORKERS)", flush=True)

    groups = [
        (participant, data_id, group)
        for (participant, data_id), group in trials.groupby(["participant_id", "dataId"], sort=True)
    ]
    if limit is not None:
        groups = groups[:limit]

    results: list[dict[str, Any]] = []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoints(checkpoint_dir)
    pending = [group for group in groups if (int(group[0]), str(group[1])) not in done]
    results.extend(done[(int(participant), str(data_id))] for participant, data_id, _ in groups
                   if (int(participant), str(data_id)) in done)
    if done and progress:
        print(f"  resuming: {len(done)} already fitted, {len(pending)} to go", flush=True)

    data_ids = sorted({data_id for _, data_id, _ in groups})
    payloads = [
        (participant, data_id, group.to_json(orient="split"))
        for participant, data_id, group in pending
    ]

    if payloads:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(data_ids, candidates, lapse_rates),
        ) as pool:
            for index, result in enumerate(pool.map(_worker_fit, payloads), start=1):
                results.append(result)
                _save_checkpoint(checkpoint_dir, result)
                if progress and (index % 10 == 0 or index == len(payloads)):
                    print(f"  fitted {index}/{len(payloads)}", flush=True)

    fits = pd.DataFrame([result["fit"] for result in results])
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    return {
        "participant_fits": fits.sort_values(["participant_id", "dataId"]).reset_index(drop=True),
        "participant_predictions": predictions,
        "condition_fits": summarise_by_condition(fits),
        "agent_comparison": build_agent_comparison(predictions),
    }


def _checkpoint_path(checkpoint_dir: Path, participant_id: Any, data_id: Any) -> Path:
    return checkpoint_dir / f"fit_{int(participant_id)}_{data_id}.json"


def _save_checkpoint(checkpoint_dir: Optional[Path], result: Mapping[str, Any]) -> None:
    if checkpoint_dir is None:
        return
    fit = result["fit"]
    path = _checkpoint_path(checkpoint_dir, fit["participant_id"], fit["dataId"])
    payload = {
        "fit": fit,
        "predictions": result["predictions"].to_json(orient="split"),
    }
    # Write beside the target and rename, so a kill mid-write cannot leave a
    # half-written checkpoint that the next run would read back as complete.
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, default=str))
    temporary.replace(path)


def _load_checkpoints(checkpoint_dir: Optional[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return {}
    done: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("fit_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        fit = payload["fit"]
        done[(int(fit["participant_id"]), str(fit["dataId"]))] = {
            "fit": fit,
            "predictions": pd.read_json(StringIO(payload["predictions"]), orient="split"),
        }
    return done


def summarise_by_condition(fits: pd.DataFrame) -> pd.DataFrame:
    """Per-condition means of the participant-level fits."""
    return (
        fits.groupby("condition", as_index=False)
        .agg(
            n_participants=("participant_id", "nunique"),
            n_trials=("n_trials", "sum"),
            nll=("nll", "mean"),
            bic=("bic", "mean"),
            accuracy_model_vs_participant=("accuracy_model_vs_participant", "mean"),
            accuracy_participant_vs_ai=("accuracy_participant_vs_ai", "mean"),
            accuracy_model_vs_ai=("accuracy_model_vs_ai", "mean"),
            decision_noise=("decision_noise", "mean"),
            memory_recall_threshold=("memory_recall_threshold", "mean"),
            opportunity_cost=("opportunity_cost", "mean"),
            lapse_rate=("lapse_rate", "mean"),
        )
        .sort_values("condition")
        .reset_index(drop=True)
    )


def build_agent_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (trial, agent), shaped like CoAX's published baselines table.

    ``Correct`` is agreement with the AI's prediction for every agent, which is
    what the task asked of participants and what CoAX's own table scores.
    """
    agents = {
        "Human": predictions["human_label"],
        "CoXAM": predictions["model_label"],
        "DT": predictions["dt_label"],
        "LR": predictions["lr_label"],
    }
    frames = []
    for agent, labels in agents.items():
        frame = predictions[
            ["participant_id", "dataId", "condition", "trial_index", "instanceId", "ai_label"]
        ].copy()
        frame["agent"] = agent
        frame["predicted"] = pd.to_numeric(labels, errors="coerce")
        frame["Correct"] = (frame["predicted"] == frame["ai_label"]).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_outputs(outputs: Mapping[str, pd.DataFrame], directory: Path, *, config: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / "run_config.json").write_text(json.dumps(config, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "server_runs" / "coxam_forward_fit")
    parser.add_argument("--workers", type=int, default=None, help=f"clamped to {MAX_WORKERS}")
    parser.add_argument("--limit", type=int, default=None, help="fit only the first N groups")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any saved checkpoints and refit every group",
    )
    args = parser.parse_args()
    checkpoint_dir = None if args.no_resume else args.output_dir / "checkpoints"

    trials = load_coxam_human_forward_trials()
    candidates = candidate_grid()
    print(
        f"CoXAM forward fit: {trials['participant_id'].nunique()} participants, "
        f"{len(trials)} trials, {len(candidates)} candidates x {len(LAPSE_RATE_VALUES)} lapse rates"
    )
    outputs = fit_participants(
        trials,
        candidates=candidates,
        n_workers=args.workers,
        limit=args.limit,
        checkpoint_dir=checkpoint_dir,
    )
    save_outputs(
        outputs,
        args.output_dir,
        config={
            "n_candidates": len(candidates),
            "lapse_rates": list(LAPSE_RATE_VALUES),
            "decision_noise": list(DECISION_NOISE_VALUES),
            "memory_recall_threshold": list(MEMORY_RECALL_THRESHOLD_VALUES),
            "opportunity_cost": list(OPPORTUNITY_COST_VALUES),
            "bic_formula": "n_parameters * log(n_trials) + 2 * total_nll",
            "task": "forward",
            "human_trials_file": str(HUMAN_TRIALS_FILE),
        },
    )
    fits = outputs["participant_fits"]
    print(f"\nfitted {len(fits)} participant-dataset groups -> {args.output_dir}")
    print(outputs["condition_fits"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Study stages, expressed only as calls into the XAIKit API layer.

Every function here is a thin adapter: it reads request parameters, fills the
gaps from the design export, calls ``xaikitTest`` (or the CoAX study runner for
a CoAX design), and serializes what came back. No experiment logic lives in the
server -- if a rule about trials, explanations or analysis needs changing, it
changes in ``src/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from src.api import xaikitTest
from src.cognitive_models import is_baseline_model_id, normalize_baseline_model_id
from src.cognitive_models.baseline_models import BASELINE_MODEL_IDS
from src.result_visualizer import plot_dv_by_two_ivs, plot_iv_dv_grid
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    coax_models_for_trials,
    run_coax_study,
)
from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
    run_coxam_study,
)

from .schemas import (
    DatasetStageRequest,
    ExplanationStageRequest,
    SimulationRequest,
    TrialsStageRequest,
)
from .serialization import (
    analysis_payload,
    design_payload,
    figure_png,
    frame_payload,
    frame_records,
    jsonable,
    report_payload,
)

#: Frameworks the ``userModel`` field of a design export can name, mapped to the
#: participant runner that serves them.
COAX_FRAMEWORKS = {"coax"}
COXAM_FRAMEWORKS = {"coxam"}

#: Frameworks the UI can select that have no runner yet. Named so the design is
#: rejected outright rather than quietly falling through to the placeholder.
UNSUPPORTED_FRAMEWORKS: set[str] = set()


def design_framework(study: xaikitTest) -> str:
    """The design's ``userModel`` value, normalized for lookup."""
    raw = str(getattr(study.design_export, "model_framework", "") or "")
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_baseline_model_id(study: xaikitTest, requested: Optional[str]) -> Optional[str]:
    """Decide which baseline the simulation should run.

    An explicit ``baseline_model_id`` on the request wins, so a caller can still
    override the design. Otherwise the design's own ``userModel`` selection is
    used, which is what the experiment-design UI writes.

    Args:
        study: The study whose design export is consulted.
        requested: The request's explicit override, if any.

    Returns:
        A canonical baseline id, or None to leave the study's model alone.

    Raises:
        ValueError: If the design names a framework with no runner behind it.
            Returning None there would silently run the placeholder stub and
            produce results that look real.
    """
    if requested:
        return requested

    framework = design_framework(study)
    if not framework:
        return None
    if is_baseline_model_id(framework):
        return framework
    if framework in UNSUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"The design selects userModel={framework!r}, which has no virtual-"
            "participant runner yet. Choose a baseline model "
            f"({', '.join(sorted(BASELINE_MODEL_IDS))}) or pass baseline_model_id "
            "explicitly."
        )
    raise ValueError(
        f"The design selects userModel={framework!r}, which is neither CoAX, "
        f"CoXAM, nor a known baseline model "
        f"({', '.join(sorted(BASELINE_MODEL_IDS))})."
    )


def build_study(design: dict[str, Any], *, project_name: str, output_dir: Path) -> xaikitTest:
    """Create the study straight from the UI export.

    ``xaikitTest(design=...)`` accepts the already-loaded dict, so the POSTed
    JSON registers every IV, CV, DV and the protocol in one step.
    """
    return xaikitTest(project_name, output_dir=output_dir, design=design)


def study_design_payload(study: xaikitTest) -> dict[str, Any]:
    """The registered design as the UI's own summary, plus design validation."""
    design = study.design_export
    ivs, cvs, dvs = study.validate_design(show=False)
    return {
        "design": design_payload(design),
        "validation": {
            "ivs": jsonable(ivs),
            "cvs": jsonable(cvs),
            "dvs": jsonable(dvs),
        },
        "support_report": report_payload(study.validate(stage="design", show=False)),
        "participant_runner": participant_runner(study),
    }


def participant_runner(study: xaikitTest) -> str:
    """Which virtual-participant runner this design's ``userModel`` selects."""
    framework = str(getattr(study.design_export, "model_framework", "") or "").strip().lower()
    if framework in COAX_FRAMEWORKS:
        return "coax"
    if framework in COXAM_FRAMEWORKS:
        return "coxam"
    return "baseline"


# -- stage 1: dataset + AI model -----------------------------------------


def run_dataset_stage(study: xaikitTest, request: DatasetStageRequest) -> dict[str, Any]:
    """Prepare the dataset and train the AI model that will be explained."""
    design = study.design_export
    dataset_id = request.dataset_id or getattr(design, "dataset_id", None)
    if not dataset_id:
        raise ValueError(
            "No dataset_id given and the design export does not name one."
        )

    data = study.prepare_dataset(
        dataset_id,
        model_type=request.model_type,
        feature_cols=request.feature_cols,
        num_features=request.num_features,
        rank_features_by_target=request.rank_features_by_target,
        test_size=request.test_size,
        random_state=request.random_state,
        show_available=False,
        show_summary=True,
    )
    study.train_AI_model(
        model_type=request.model_type,
        target_metric=request.target_metric,
        target_score=request.target_score,
        max_epochs=request.max_epochs,
        check_every_epochs=request.check_every_epochs,
        batch_size=request.batch_size,
        verbose=False,
    )
    study.evaluate(split="both")

    return {
        "dataset": {
            "dataset_id": data.dataset_id,
            "feature_names": list(data.feature_names),
            "model_feature_names": list(data.model_feature_names),
            "n_train": int(len(data.y_train)),
            "n_test": int(len(data.y_test)),
        },
        "model": {
            "model_name": study.model_name,
            "test_accuracy": float(study.test_accuracy()),
            "training_summary": frame_records(study.training_summary_table()),
            "training_history": frame_records(study.training_history_table()),
            "metrics": frame_records(study.metrics_table().reset_index()),
            "confusion_matrix": frame_records(study.confusion_matrix_table().reset_index()),
        },
    }


# -- stage 2: trials ------------------------------------------------------


def run_trials_stage(study: xaikitTest, request: TrialsStageRequest) -> dict[str, Any]:
    """Generate the balanced trial table the participants will be run over."""
    design = study.design_export
    participants = (
        request.participants_per_between_condition
        if request.participants_per_between_condition is not None
        else getattr(design, "participants_per_condition", None)
    )
    if participants is None:
        raise ValueError(
            "participants_per_between_condition was not given and the design "
            "export does not record participants per condition."
        )

    num_training = request.num_training
    num_testing = request.num_testing
    if num_testing is None:
        total = getattr(design, "trials_per_participant", None)
        if total is None:
            raise ValueError(
                "num_testing was not given and the design export does not record "
                "trials per participant."
            )
        # The export records one total; the training/testing split is the
        # server's parameter, exactly as the tutorials state it explicitly.
        num_testing = int(total) - int(num_training)
        if num_testing <= 0:
            raise ValueError(
                f"num_training={num_training} leaves no testing trials out of the "
                f"{total} trials per participant the design records."
            )

    result = study.generate_trials(
        participants_per_between_condition=int(participants),
        num_training=int(num_training),
        num_testing=int(num_testing),
        balance_by_ai_prediction=request.balance_by_ai_prediction,
        seed=request.seed,
        output_dir=request.output_dir,
        preview_rows=0,
        show=False,
    )
    trials = pd.DataFrame(result.trials)

    condition_columns = [
        column
        for column in (iv["name"] for iv in getattr(design, "ivs", []))
        if column in trials.columns
    ]
    return {
        "counts": {
            "trials": int(len(trials)),
            "participants": int(trials["participantId"].nunique()),
            "training": int((trials["phase"] == "training").sum()),
            "testing": int((trials["phase"] == "testing").sum()),
        },
        "num_training": int(num_training),
        "num_testing": int(num_testing),
        "participants_per_between_condition": int(participants),
        "by_condition": (
            frame_records(trials.groupby(condition_columns).size().reset_index(name="trials"))
            if condition_columns
            else []
        ),
        "preview": frame_records(trials, limit=request.preview_rows),
        "files": {"csv": result.csv_path, "json": result.json_path, "summary": result.summary_path},
    }


# -- stage 3: explanations ------------------------------------------------


def run_explanations_stage(study: xaikitTest, request: ExplanationStageRequest) -> dict[str, Any]:
    """Generate one XAI table per method in the design, plus AI predictions."""
    path, pool = study.explanations(
        methods=request.methods,
        output_dir=request.output_dir,
        target=request.target,
        method_kwargs=request.method_kwargs,
        show_checks=True,
    )
    counts = pool["expMethod"].value_counts().rename_axis("expMethod").reset_index(name="rows")
    return {
        "combined_table": str(path),
        "rows": int(len(pool)),
        "by_method": frame_records(counts),
        "methods": sorted(
            method for method in pool["expMethod"].astype(str).unique()
            if method != "__prediction_only__"
        ),
        "files": [str(item) for item in study.explanation_paths],
    }


# -- stage 4: simulation --------------------------------------------------


def run_simulation_stage(
    study: xaikitTest,
    request: SimulationRequest,
    *,
    output_subdir: str,
) -> dict[str, Any]:
    """Run the virtual participants and save the run as CSV and JSON.

    ``mode`` is the API layer's own selection vocabulary, so the same endpoint
    serves a single-trial walkthrough and the full experiment:
    ``trial_by_trial``, ``participant_by_participant``, ``whole_condition``
    (with ``condition_filter``) and ``whole_experiment``.
    """
    runner = participant_runner(study)
    if runner == "coax":
        results = run_coax_study(
            study,
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            cognitive_model=coax_models_for_trials(
                pd.DataFrame(study.trials),
                strategies=request.coax_strategies,
                params=request.coax_params,
            ),
            store=True,
        )
    elif runner == "coxam":
        results = run_coxam_study(
            study,
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            source=request.coxam_source,
            policy_override=request.coxam_policy,
            eval_params=request.coxam_eval_params,
            store=True,
        )
    else:
        baseline_model_id = resolve_baseline_model_id(study, request.baseline_model_id)
        if baseline_model_id:
            study.set_cognitive_model(
                cognitive_model_id=baseline_model_id,
                model_kwargs=request.baseline_model_kwargs,
            )
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
        )

    if results.empty:
        raise ValueError(
            f"mode={request.mode!r} selected no trials. Check participant_id and "
            "condition_filter against the generated trial table."
        )

    csv_path, json_path = study.save_results(out_dir=output_subdir)
    testing = results[results["phase"].astype(str) == "testing"] if "phase" in results else results
    return {
        "runner": runner,
        "cognitive_model_id": study.cognitive_model_id,
        "mode": request.mode,
        "participant_id": request.participant_id,
        "condition_filter": jsonable(request.condition_filter),
        "counts": {
            "steps": int(len(results)),
            "testing_steps": int(len(testing)),
            "participants": (
                int(results["participantId"].nunique())
                if "participantId" in results.columns
                else None
            ),
        },
        "by_step": (
            frame_records(results.groupby(["phase", "step"]).size().reset_index(name="rows"))
            if {"phase", "step"} <= set(results.columns)
            else []
        ),
        "preview": frame_records(results, limit=request.preview_rows),
        "files": {"csv": csv_path, "json": json_path},
    }


# -- results, analysis, plots --------------------------------------------


def require_results(study: xaikitTest) -> pd.DataFrame:
    if study.simulated_results is None:
        raise ValueError("No simulation has been run for this study yet.")
    return study.simulated_results


def results_payload(
    study: xaikitTest,
    *,
    phase: Optional[str] = None,
    participant_id: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Trial-by-trial step rows, filtered and paged for step-through rendering."""
    results = _filter_results(require_results(study), phase=phase, participant_id=participant_id)
    return frame_payload(results, limit=limit, offset=offset)


def results_csv(
    study: xaikitTest,
    *,
    phase: Optional[str] = None,
    participant_id: Optional[int] = None,
) -> str:
    """The same rows as CSV text, for a browser download."""
    results = _filter_results(require_results(study), phase=phase, participant_id=participant_id)
    return results.to_csv(index=False)


def _filter_results(
    results: pd.DataFrame,
    *,
    phase: Optional[str],
    participant_id: Optional[int],
) -> pd.DataFrame:
    if phase and "phase" in results.columns:
        results = results[results["phase"].astype(str).str.lower() == phase.lower()]
    if participant_id is not None and "participantId" in results.columns:
        results = results[results["participantId"].astype(int) == int(participant_id)]
    return results


def analysis_for(
    study: xaikitTest,
    *,
    ivs: Optional[Sequence[str]] = None,
    dvs: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Every requested IV x DV analysis, defaulting to the design's own lists."""
    require_results(study)
    design = study.design_export
    ivs = list(ivs) if ivs else [iv["name"] for iv in getattr(design, "between_ivs", [])]
    dvs = list(dvs) if dvs else list(getattr(design, "simulatable_dvs", []))
    if not ivs or not dvs:
        raise ValueError(
            "No IV/DV pair to analyze. Pass ivs and dvs explicitly -- the design "
            "export has no between-subject IVs or no simulatable DVs."
        )

    analyses = []
    skipped = []
    for dv in dvs:
        for iv in ivs:
            try:
                analyses.append(analysis_payload(study.analyze_iv_dv(iv=iv, dv=dv)))
            except (ValueError, KeyError) as error:
                # One unanalysable pair (a DV the runner never filled, an IV with
                # a single level after filtering) must not fail the whole request.
                skipped.append({"iv": iv, "dv": dv, "reason": str(error)})
    return {"ivs": ivs, "dvs": dvs, "analyses": analyses, "skipped": skipped}


def interaction_plot_payload(
    study: xaikitTest,
    *,
    x_iv: str,
    hue_iv: str,
    dv: str,
    phase: Optional[str] = "testing",
    errorbar: Optional[str] = "sem",
    title: Optional[str] = None,
    include_png: bool = False,
) -> dict[str, Any]:
    """Aggregated bars for one DV against two IVs, as data the UI can draw."""
    plot = plot_dv_by_two_ivs(
        require_results(study),
        x_iv=x_iv,
        hue_iv=hue_iv,
        dv=dv,
        phase=phase,
        errorbar=errorbar,
        title=title,
    )
    return _plot_payload(
        plot.summary,
        plot.figure,
        include_png=include_png,
        spec={"kind": "interaction", "x_iv": x_iv, "hue_iv": hue_iv, "dv": dv, "phase": phase},
    )


def grid_plot_payload(
    study: xaikitTest,
    *,
    ivs: Optional[Sequence[str]] = None,
    dvs: Optional[Sequence[str]] = None,
    phase: Optional[str] = "testing",
    errorbar: Optional[str] = "sem",
    title: Optional[str] = "Experiment results",
    include_png: bool = False,
) -> dict[str, Any]:
    """Aggregated bars for every DV x IV combination in the design."""
    design = study.design_export
    ivs = list(ivs) if ivs else [iv["name"] for iv in getattr(design, "ivs", [])]
    dvs = list(dvs) if dvs else list(getattr(design, "simulatable_dvs", []))
    grid = plot_iv_dv_grid(
        require_results(study),
        ivs=ivs,
        dvs=dvs,
        phase=phase,
        errorbar=errorbar,
        title=title,
    )
    return _plot_payload(
        grid.summary,
        grid.figure,
        include_png=include_png,
        spec={"kind": "grid", "ivs": ivs, "dvs": dvs, "phase": phase},
    )


def _plot_payload(
    summary: pd.DataFrame,
    figure: Any,
    *,
    include_png: bool,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Send the aggregated values; the PNG only when the client asks for it.

    The plot helpers return the participant-level aggregate they drew, so the
    UI can render the same numbers with its own chart library instead of
    embedding a server-rendered image.
    """
    import matplotlib.pyplot as plt

    payload = {"spec": spec, "summary": frame_records(summary)}
    if include_png:
        payload["png"] = figure_png(figure)
    else:
        plt.close(figure)
    return payload

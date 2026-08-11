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

import numpy as np
import pandas as pd

from src.api import xaikitTest
from src.cognitive_models import is_baseline_model_id, normalize_baseline_model_id
from src.experiment_planner.design_export import normalize_cognitive_params
from src.cognitive_models.baseline_models import BASELINE_MODEL_IDS
from src.result_visualizer import plot_dv_by_two_ivs, plot_iv_dv_grid
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    coax_models_for_trials,
)
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_study_runner import (
    run_sim2real_study,
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
    posthoc_payload,
    report_payload,
)

#: Frameworks the ``userModel`` field of a design export can name, mapped to the
#: participant runner that serves them.
COAX_FRAMEWORKS = {"coax"}
COXAM_FRAMEWORKS = {"coxam"}
SIM2REAL_FRAMEWORKS = {"sim2real"}

#: Frameworks the UI can select that have no runner yet. Named so the design is
#: rejected outright rather than quietly falling through to the placeholder.
UNSUPPORTED_FRAMEWORKS: set[str] = set()


def design_framework(study: xaikitTest) -> str:
    """Which agent this design actually runs, as a canonical id.

    Delegates to ``DesignExport.resolved_framework`` rather than re-deriving it
    from the raw ``userModel`` string. That matters for two real UI values:
    ``userModel: "CoAX"`` on a design that varies ``xai_property`` is a Sim2Real
    study (its cognitive model is a CoAX-derived attribution sum, but it runs
    the Sim2Real fitted model), and ``userModel: "CoAX (XAI Property)"`` slugifies
    to ``coax_xai_property`` -- matching neither "coax" nor "sim2real" -- so both
    would silently fall through to the generic baseline runner without this.
    """
    design = getattr(study, "design_export", None)
    resolved = getattr(design, "resolved_framework", "") if design is not None else ""
    if resolved:
        return str(resolved)
    raw = str(getattr(design, "model_framework", "") or "")
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
    if not framework or framework in COAX_FRAMEWORKS | COXAM_FRAMEWORKS | SIM2REAL_FRAMEWORKS:
        # Has a real participant runner; nothing for the baseline path to resolve.
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
    framework = design_framework(study)
    if framework in COAX_FRAMEWORKS:
        return "coax"
    if framework in COXAM_FRAMEWORKS:
        return "coxam"
    if framework in SIM2REAL_FRAMEWORKS:
        return "sim2real"
    return "baseline"


# -- stage 1: dataset + AI model -----------------------------------------


def _coxam_corpus_covers(dataset_id: str) -> bool:
    """Whether CoXAM shipped a published corpus for this dataset."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        COXAM_CORPUS_FEATURES,
    )

    return dataset_id in COXAM_CORPUS_FEATURES


#: Datasets CoAX's published user-study corpus (assets/ai_dataset/CoAX/none.csv)
#: carries AI predictions for -- confirmed against the corpus file itself, and
#: matching coax_human_replay.FITTED_DATA_IDS for the same underlying reason:
#: mushrooms was collected but never trained/fitted for CoAX.
COAX_CORPUS_DATA_IDS = frozenset({"adult", "forest_cover", "wine_quality"})


def _coax_corpus_covers(dataset_id: str) -> bool:
    """Whether CoAX shipped a published corpus for this dataset."""
    return dataset_id in COAX_CORPUS_DATA_IDS


def _resolve_dataset_ids(request: DatasetStageRequest, design: Any) -> list[str]:
    """Every dataset this stage should prepare, in declared order.

    An explicit ``request.dataset_id`` (scalar or list) always wins. Otherwise
    ``design.dataset_ids`` already resolves both shapes the export can name a
    dataset with -- the older singular ``studyDesign.dataset`` field, or a
    between-subjects IV named ``dataset`` with 2+ levels -- so nothing further
    needs deciding here.
    """
    if request.dataset_id:
        return (
            list(request.dataset_id)
            if isinstance(request.dataset_id, (list, tuple))
            else [request.dataset_id]
        )
    return list(getattr(design, "dataset_ids", None) or [])


def _dataset_payload_entry(data: Any) -> dict[str, Any]:
    return {
        "dataset_id": data.dataset_id,
        "feature_names": list(data.feature_names),
        "model_feature_names": list(data.model_feature_names),
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
    }


def _corpus_skip_reason(framework: str, dataset_id: str) -> Optional[str]:
    """Why AI training can be skipped for this framework/dataset, or None."""
    if framework == "sim2real":
        return (
            "Sim2Real simulates from a fixed published corpus and never reads "
            "trained_ai_model, so no AI model is trained here."
        )
    if framework == "coxam" and _coxam_corpus_covers(dataset_id):
        return (
            f"CoXAM's published corpus covers {dataset_id!r}: "
            "run_coxam_study(source='assets') reads its own AI predictions and "
            "DT/LR surrogates, never trained_ai_model, so no AI model is "
            "trained here. Pass a dataset the corpus does not cover, or use a "
            "different agent, to train one."
        )
    if framework == "coax" and _coax_corpus_covers(dataset_id):
        return (
            f"CoAX's published corpus covers {dataset_id!r}: "
            "run_coax_study(source='corpus') reads its own AI predictions and "
            "explanation vectors, never trained_ai_model, so no AI model is "
            "trained here. Pass a dataset the corpus does not cover to train one."
        )
    return None


def _finish_dataset_stage(
    study: xaikitTest,
    request: DatasetStageRequest,
    dataset_payload: dict[str, Any],
    skip_reason: Optional[str],
) -> dict[str, Any]:
    if skip_reason:
        # train_AI_model would normally set this; run_coxam_study(source="assets")
        # looks the corpus's AI predictions up by model_name, so a request for
        # instances "with a labelled AI prediction" would otherwise search for
        # None/a model that was never trained instead of the corpus's own "mlp".
        study.model_name = request.model_type
        dataset_payload["model"] = None
        dataset_payload["model_skipped_reason"] = skip_reason
        return dataset_payload

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

    dataset_payload["model"] = {
        "model_name": study.model_name,
        "test_accuracy": float(study.test_accuracy()),
        "training_summary": frame_records(study.training_summary_table()),
        "training_history": frame_records(study.training_history_table()),
        "metrics": frame_records(study.metrics_table().reset_index()),
        "confusion_matrix": frame_records(study.confusion_matrix_table().reset_index()),
    }
    return dataset_payload


def run_dataset_stage(study: xaikitTest, request: DatasetStageRequest) -> dict[str, Any]:
    """Prepare the dataset(s) and, where a published corpus makes it
    unnecessary, skip training the AI model it would otherwise explain.

    ``generate_trials()`` needs prepared data regardless of participant
    runner, so this always calls ``prepare_dataset``. Training is skipped for:

    * **Sim2Real**, always -- its simulation reads a fixed published corpus,
      not ``study.trained_ai_model`` (see ``run_sim2real_study``'s own
      docstring). Sim2Real also does not support a multi-dataset study at all.
    * **CoXAM**, when the dataset is one its corpus covers -- ``prepare_dataset``
      already routed onto that corpus's own feature set (see
      ``xaikitTest.prepare_dataset``'s ``cognitive_model_id``), and
      ``run_coxam_study(source="assets")`` reads the corpus's own AI
      predictions and DT/LR surrogates, never ``trained_ai_model``. Training
      here would be a full MLP run spent on a model nothing downstream reads.
      A CoXAM dataset the corpus does *not* cover still trains, since only
      ``source="fit"`` can serve it.
    * **A multi-dataset study**, one dataset at a time is never trained here --
      ``study.train_AI_model()`` only ever has one ``study.data`` to train
      against. Every requested dataset must therefore already be corpus-
      covered (source='assets'/'corpus' can serve all of them), or this
      raises rather than silently training against just one of them.
    """
    design = study.design_export
    dataset_ids = _resolve_dataset_ids(request, design)
    if not dataset_ids:
        raise ValueError(
            "No dataset_id given and the design export does not name one."
        )

    framework = design_framework(study)

    if len(dataset_ids) == 1:
        data = study.prepare_dataset(
            dataset_ids[0],
            model_type=request.model_type,
            feature_cols=request.feature_cols,
            num_features=request.num_features,
            rank_features_by_target=request.rank_features_by_target,
            test_size=request.test_size,
            random_state=request.random_state,
            show_available=False,
            show_summary=True,
            cognitive_model_id=framework,
        )
        dataset_payload = {"dataset": _dataset_payload_entry(data)}
        skip_reason = _corpus_skip_reason(framework, data.dataset_id)
        return _finish_dataset_stage(study, request, dataset_payload, skip_reason)

    # -- multi-dataset, between-subjects ------------------------------------
    if framework == "sim2real":
        raise ValueError(
            f"Sim2Real does not support a multi-dataset study; got {dataset_ids!r}."
        )

    uncovered = [
        dataset_id for dataset_id in dataset_ids
        if _corpus_skip_reason(framework, dataset_id) is None
    ]
    if uncovered:
        raise ValueError(
            "Multi-dataset AI training is not yet supported -- each dataset's "
            f"model can only be trained one at a time, but dataset(s) {uncovered!r} "
            f"have no published {framework} corpus to run source='assets'/'corpus' "
            "against instead. Remove them from the study, or prepare/train each "
            "uncovered dataset on its own with a single dataset_id first."
        )

    data_by_dataset = study.prepare_dataset(
        dataset_ids,
        model_type=request.model_type,
        feature_cols=request.feature_cols,
        num_features=request.num_features,
        rank_features_by_target=request.rank_features_by_target,
        test_size=request.test_size,
        random_state=request.random_state,
        show_available=False,
        show_summary=True,
        cognitive_model_id=framework,
    )
    dataset_payload = {
        "datasets": {
            dataset_id: _dataset_payload_entry(data)
            for dataset_id, data in data_by_dataset.items()
        }
    }
    skip_reason = (
        f"Multi-dataset study: every dataset ({', '.join(dataset_ids)}) is "
        f"covered by {framework}'s published corpus, so no AI model is trained "
        "-- each dataset's own corpus predictions/surrogates are used at "
        "simulation time instead."
    )
    return _finish_dataset_stage(study, request, dataset_payload, skip_reason)


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

    total_trials = getattr(design, "trials_per_participant", None)
    apparatus_test_count = len(getattr(design, "apparatus_instance_ids", []) or [])

    num_training = request.num_training
    if num_training is None:
        if total_trials is not None and apparatus_test_count:
            # Derive from the design, not a flat guess: 30 trials/participant
            # with 20 apparatus testing instances means 10 training trials,
            # not an arbitrary constant that happens to leave 26 "testing"
            # trials -- more than the apparatus declared testing instances,
            # which is exactly the mismatch that surfaced this.
            num_training = max(0, int(total_trials) - apparatus_test_count)
        else:
            num_training = 4  # no apparatus/total to derive from; old default

    num_testing = request.num_testing
    if num_testing is None:
        if total_trials is None:
            raise ValueError(
                "num_testing was not given and the design export does not record "
                "trials per participant."
            )
        # The export records one total; the training/testing split is the
        # server's parameter, exactly as the tutorials state it explicitly.
        num_testing = int(total_trials) - int(num_training)
        if num_testing <= 0:
            raise ValueError(
                f"num_training={num_training} leaves no testing trials out of the "
                f"{total_trials} trials per participant the design records."
            )

    balance_by_ai_prediction = request.balance_by_ai_prediction
    if balance_by_ai_prediction is None:
        # A model exists only if the dataset stage trained one -- Sim2Real
        # designs skip that (see run_dataset_stage), so this follows what
        # actually happened rather than assuming every design trained a model.
        balance_by_ai_prediction = getattr(study, "trained_ai_model", None) is not None

    apparatus_test_ids = sorted(set(getattr(design, "apparatus_instance_ids", [])))
    split_overridden = bool(apparatus_test_ids)
    data_by_dataset = getattr(study, "data_by_dataset", None)
    if apparatus_test_ids and isinstance(data_by_dataset, dict) and data_by_dataset:
        # Multi-dataset: the exact same override, applied once per dataset
        # from that dataset's own per-appId apparatus declarations -- never
        # applying one dataset's declared instance ids to another's split.
        ids_by_dataset = getattr(design, "apparatus_instance_ids_by_dataset", {}) or {}
        training_ids_by_dataset = getattr(design, "apparatus_training_instance_ids_by_dataset", {}) or {}
        for dataset_id, data in data_by_dataset.items():
            dataset_test_ids = sorted(set(ids_by_dataset.get(dataset_id, [])))
            if not dataset_test_ids:
                continue
            declared_train_ids = sorted(set(training_ids_by_dataset.get(dataset_id, [])))
            if declared_train_ids:
                train_ids = declared_train_ids
            elif num_training > 0:
                pool = (
                    set(data.split.train_instance_ids.tolist())
                    | set(data.split.test_instance_ids.tolist())
                ) - set(dataset_test_ids)
                if design_framework(study) == "coxam":
                    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
                        coxam_available_instance_ids,
                    )

                    pool &= set(coxam_available_instance_ids(dataset_id))
                pool = sorted(pool)
                if len(pool) < num_training:
                    raise ValueError(
                        f"Dataset {dataset_id!r}'s apparatus declares {len(dataset_test_ids)} "
                        f"testing instances; only {len(pool)} remain for the {num_training} "
                        "training instances requested, with no overlap allowed. Declare "
                        "trainingInstanceIds in that dataset's apparatus configuration, "
                        "lower num_training, or widen instanceIds."
                    )
                rng = np.random.default_rng(request.seed)
                train_ids = sorted(int(i) for i in rng.choice(pool, size=num_training, replace=False))
            else:
                train_ids = []
            data.split.test_instance_ids = np.asarray(dataset_test_ids)
            data.split.train_instance_ids = np.asarray(train_ids)
    elif apparatus_test_ids:
        # The apparatus declares exactly which instances were tested; the
        # dataset's own train/test split was drawn independently (by
        # prepare_dataset's test_size/random_state) and has no reason to
        # agree with it. Filtering that split down to the allowed set -- what
        # the code below does for a design with no apparatus -- silently kept
        # only whichever fraction of the 20 declared instances happened to
        # land in the split's own test portion (10 of 20 for this design),
        # so the split is overridden outright instead: test = exactly what
        # the apparatus declared, train = its own declared instances if any,
        # else a random sample of whatever is left, never overlapping test.
        declared_train_ids = sorted(set(getattr(design, "apparatus_training_instance_ids", [])))
        if declared_train_ids:
            train_ids = declared_train_ids
        elif num_training > 0:
            pool = (
                set(study.data.split.train_instance_ids.tolist())
                | set(study.data.split.test_instance_ids.tolist())
            ) - set(apparatus_test_ids)
            if design_framework(study) == "coxam":
                # Sampling outside the published corpus is worse than useless
                # here: source="assets" can only serve what it shipped, so an
                # id the raw dataset has but the corpus does not would fail
                # downstream instead of simulating.
                from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
                    coxam_available_instance_ids,
                )

                pool &= set(coxam_available_instance_ids(study.data.dataset_id))
            pool = sorted(pool)
            if len(pool) < num_training:
                raise ValueError(
                    f"The apparatus declares {len(apparatus_test_ids)} testing instances; "
                    f"only {len(pool)} remain for the {num_training} training instances "
                    "requested, with no overlap allowed. Declare trainingInstanceIds in "
                    "the apparatus, lower num_training, or widen instanceIds."
                )
            rng = np.random.default_rng(request.seed)
            train_ids = sorted(int(i) for i in rng.choice(pool, size=num_training, replace=False))
        else:
            train_ids = []
        study.data.split.test_instance_ids = np.asarray(apparatus_test_ids)
        study.data.split.train_instance_ids = np.asarray(train_ids)

    allowed_instance_ids = request.allowed_instance_ids
    if allowed_instance_ids is None and not split_overridden:
        # Union, not just the testing set: generate_trials takes one pool for
        # both phases, and a training instance the apparatus declared is still
        # one a human participant was shown. Skipped when the split was
        # already overridden above: that already set the train/test pools to
        # exactly the right ids (including a randomly-sampled training id
        # outside the apparatus's declared set), and this filter would just
        # discard that sample again.
        allowed_instance_ids = sorted(
            set(getattr(design, "apparatus_instance_ids", []))
            | set(getattr(design, "apparatus_training_instance_ids", []))
        ) or None

    result = study.generate_trials(
        participants_per_between_condition=int(participants),
        num_training=int(num_training),
        num_testing=int(num_testing),
        balance_by_ai_prediction=balance_by_ai_prediction,
        allowed_instance_ids=allowed_instance_ids,
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
    """Generate one XAI table per method in the design, plus AI predictions.

    A no-op for CoXAM and Sim2Real: neither ever reads what this stage
    produces (``study.combined_explanations``). CoXAM's ``run_coxam_study``
    builds its own DT/LR surrogates internally (from a trained model, or from
    the published corpus); Sim2Real's ``run_sim2real_study`` reads a fixed
    published corpus via ``Sim2RealAttributionProjector.from_assets()`` and
    never fits anything from a study's own data. Calling ``study.explanations()``
    needs LIME/SHAP to explain a real ``trained_ai_model``, so for a design
    whose dataset stage skipped training (see ``run_dataset_stage`` --
    Sim2Real always skips it, CoXAM skips it for a corpus-covered dataset) it
    always raised -- a UI that calls every stage in sequence regardless of
    agent should not have to know which agents to skip this one for.

    Also a no-op for CoAX specifically when its dataset stage skipped training
    (a corpus-covered dataset -- see ``_coax_corpus_covers``):
    ``run_coax_study(source='corpus')`` reads the published CoAX corpus's own
    explanation vectors directly and never touches ``combined_explanations``
    either. A CoAX dataset the corpus does *not* cover still needs this stage,
    since only ``source='study'`` can serve it.
    """
    framework = design_framework(study)
    if framework in {"coxam", "sim2real"} or (
        framework == "coax" and getattr(study, "trained_ai_model", None) is None
    ):
        return {
            "combined_table": None,
            "rows": 0,
            "by_method": [],
            "methods": [],
            "files": [],
            "skipped_reason": (
                "CoXAM builds its own DT/LR surrogates inside run_coxam_study, "
                "Sim2Real reads a fixed published corpus inside run_sim2real_study, "
                "and this CoAX dataset is served by run_coax_study(source='corpus') -- "
                "none of these read this stage's output, so nothing is generated here."
            ),
        }

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
        "skipped_reason": None,
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

    coax/coxam are dispatched through ``study.run_experiment(...)`` rather
    than calling ``run_coax_study``/``run_coxam_study`` directly -- the two
    used to be equivalent, but ``run_experiment`` is also where a
    multi-dataset study's per-dataset simulation loop lives
    (``xaikitTest._run_multi_dataset_experiment``): calling the runner
    directly would bypass that loop entirely and crash on ``study.data is
    None`` for a multi-dataset study. ``store=True`` is not passed here --
    ``run_experiment`` already hardcodes it for these two agents.
    """
    runner = participant_runner(study)
    if runner == "coax":
        coax_source = request.coax_source
        if coax_source is None:
            # source="study" needs trained_ai_model; the dataset stage only
            # trains one when the corpus does not already cover this dataset
            # (see run_dataset_stage), so this follows what actually happened
            # rather than assuming every design trained a model.
            coax_source = "study" if getattr(study, "trained_ai_model", None) is not None else "corpus"
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            cognitive_model=coax_models_for_trials(
                pd.DataFrame(study.trials),
                strategies=request.coax_strategies,
                params=request.coax_params,
            ),
            source=coax_source,
        )
    elif runner == "coxam":
        coxam_source = request.coxam_source
        if coxam_source is None:
            # 'fit' needs trained_ai_model; the dataset stage only trains one
            # when the corpus does not already cover this dataset (see
            # run_dataset_stage), so this follows what actually happened
            # rather than assuming every design trained a model.
            coxam_source = "fit" if getattr(study, "trained_ai_model", None) is not None else "assets"
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            source=coxam_source,
            policy_override=request.coxam_policy,
            eval_params=normalize_cognitive_params("coxam", request.coxam_eval_params),
        )
    elif runner == "sim2real":
        # Counterfactual-only, and its corpus supplies its own stimuli and
        # ground truth, so it needs neither a task switch nor a trained model.
        results = run_sim2real_study(
            study,
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            strategy=request.sim2real_strategy,
            cognitive_params=normalize_cognitive_params("sim2real", request.sim2real_params),
            normalize_by_i_max=request.sim2real_normalize_by_i_max,
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


def posthoc_for(
    study: xaikitTest,
    *,
    dv: str,
    condition_cols: Optional[Sequence[str]] = None,
    correction: Optional[str] = "holm",
    phase: str = "testing",
) -> dict[str, Any]:
    """Every pairwise condition comparison for one DV, with corrected p-values.

    Complements ``analysis_for``'s omnibus per-IV test: this compares every
    crossed condition cell against every other (e.g. Rules-with-XAI vs
    Weights-without-XAI), which is what a bar plot with significance
    annotations needs. Defaults ``condition_cols`` to every between- and
    within-subject IV the design declares.
    """
    require_results(study)
    design = study.design_export
    condition_cols = (
        list(condition_cols)
        if condition_cols
        else [iv["name"] for iv in getattr(design, "ivs", [])]
    )
    if not condition_cols:
        raise ValueError(
            "No condition columns to compare. Pass condition_cols explicitly -- "
            "the design export declares no IVs."
        )
    result = study.pairwise_condition_tests(
        dv=dv,
        condition_cols=condition_cols,
        correction=correction,
        phase=phase,
    )
    from src.result_visualizer import pretty

    payload = posthoc_payload(result)
    payload["dv"] = dv
    payload["dv_label"] = pretty(dv)
    payload["condition_cols"] = condition_cols
    payload["condition_cols_label"] = [pretty(column) for column in condition_cols]
    return payload


def interaction_plot_payload(
    study: xaikitTest,
    *,
    x_iv: str,
    hue_iv: str,
    dv: str,
    phase: Optional[str] = "testing",
    errorbar: Optional[str] = "ci95",
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
    errorbar: Optional[str] = "ci95",
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

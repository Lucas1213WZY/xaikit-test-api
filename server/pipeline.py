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
from src.virtual_experiment_executor.participant_pools import is_diverse_mode
from src.experiment_planner.design_export import normalize_cognitive_params
from src.cognitive_models.baseline_models import BASELINE_MODEL_IDS
from src.result_visualizer import plot_dv_by_two_ivs, plot_iv_dv_grid
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    coax_models_for_trials,
)
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    COAX_STRATEGIES_BY_XAI_TYPE,
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


def _as_dict(value: Any) -> dict[Any, Any]:
    """``value`` if it is a real dict, else ``{}``.

    ``study`` is a bare ``MagicMock()`` in many tests -- an unconfigured
    attribute access (``study.data_by_dataset``) auto-vivifies a *truthy*
    ``MagicMock`` rather than the real empty dict a fresh ``xaikitTest``
    instance would have, so a plain truthiness check reads a mocked
    single-dataset study as multi-dataset. Matches the ``isinstance(...,
    dict)`` guard already used elsewhere in this module for the same reason.
    """
    return value if isinstance(value, dict) else {}


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


def _coax_corpus_methods(dataset_id: str) -> set[str]:
    """``xai_method`` values the published CoAX corpus actually collected for
    one dataset.

    The corpus is fixed, pre-collected human-study data -- it can only ever
    replay the method(s) it was collected with (one per dataset today: lime
    for adult/wine_quality, shap for forest_cover), never a method a design's
    ``xai_method`` IV asks for but the corpus never ran.
    """
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
        load_coax_corpus_tables,
    )

    rows = load_coax_corpus_tables()["attribution"]
    rows = rows[rows["dataId"].astype(str) == str(dataset_id)]
    return set(rows["expMethod"].astype(str).unique())


def _design_xai_methods(design: Any) -> list[str]:
    """The design's declared ``xai_method`` IV levels, or ``[]`` if it has none."""
    for iv in getattr(design, "ivs", None) or []:
        if iv.get("name") == "xai_method":
            return list(iv.get("levels") or [])
    return []


def _coax_needs_real_generation(design: Any, framework: str, dataset_id: str) -> bool:
    """Whether this dataset's declared ``xai_method`` levels force real CoAX
    training and explanation generation instead of the published corpus.

    Routing to ``source='corpus'`` for a design that also asks for a method
    the corpus never collected silently builds an explanation repository
    with nothing in it for that method -- the failure then only surfaces much
    later, deep in trial execution, with a message that names neither the
    dataset nor the method at fault (see ``build_coax_study_repository``).
    Forcing real training+generation here -- the same override
    ``needs_baselines`` already applies -- catches it immediately instead,
    either by actually generating the missing method(s) (single dataset) or
    by raising the existing, clearer "multi-dataset training is not yet
    supported" error (multi-dataset), rather than the cryptic one.
    """
    if framework != "coax":
        return False
    declared = _design_xai_methods(design)
    if not declared:
        return False
    return not set(declared) <= (_coax_corpus_methods(dataset_id) | {"none"})


def _coxam_needs_real_generation(design: Any, framework: str) -> bool:
    """Whether this design's counterfactual task forces real CoXAM training,
    overriding a corpus-covered dataset's usual skip.

    CoXAM's published corpus stores fitted DT/LR surrogates and static
    predictions for a fixed instance set -- ``run_coxam_study(source="assets")``
    reads those directly for forward simulation, but counterfactual simulation
    evaluates arbitrary participant-proposed feature perturbations against the
    real AI model (see ``counterfactual_env.py``'s ``predict_fn``), which no
    static corpus table can answer. A design whose DVs/``user_task`` ask for
    the counterfactual task needs a real ``trained_ai_model`` even on a dataset
    CoXAM's corpus otherwise covers -- skipping training here would let a later
    ``/simulate`` call for that task crash on a ``None`` model instead of
    surfacing the cause at the dataset stage.
    """
    if framework != "coxam":
        return False
    return "counterfactual_simulation" in getattr(design, "user_tasks", [])


def _coax_baseline_servable_from_corpus(design: Any, framework: str, dataset_id: str) -> bool:
    """Whether a declared ``mlProxyBaselines`` model can read this dataset's
    published CoAX corpus directly instead of needing a real trained model.

    A baseline (KNN, decision tree, ...) is a machine-learning proxy reading
    predictions/explanations/features to stand in for a human -- it does not
    explain the AI model itself, so it needs exactly what
    ``run_coax_study(source='corpus')`` already reads for the primary agent,
    nothing more. ``combined_explanations_from_corpus`` reshapes that same
    corpus data into the shape the generic executor's baseline path reads.
    CoAX-only for now -- CoXAM's/Sim2Real's baseline+corpus interaction is
    unchanged, still forcing real training/generation when a baseline is
    declared alongside them.
    """
    if framework != "coax":
        return False
    return _corpus_skip_reason(framework, dataset_id) is not None and not _coax_needs_real_generation(
        design, framework, dataset_id
    )


def _corpus_instance_ids(framework: str, dataset_id: str) -> Optional[list[int]]:
    """Instance ids the published corpus can serve for this framework/dataset,
    or None when neither applies (an uncovered dataset, or a framework with no
    corpus concept at all).

    Used as the trial-generation default when nothing else (apparatus
    instanceIds, an explicit allowed_instance_ids) restricts a corpus-covered
    dataset's pool -- without it, trials sample freely from the dataset's own
    random split and later fail at simulation time the first time they
    reference an instance the corpus never shipped.
    """
    if framework == "coxam" and _coxam_corpus_covers(dataset_id):
        from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
            coxam_available_instance_ids,
        )

        return coxam_available_instance_ids(dataset_id)
    if framework == "coax" and _coax_corpus_covers(dataset_id):
        from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
            coax_available_instance_ids,
        )

        return coax_available_instance_ids(dataset_id)
    return None


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


#: Accuracy-shaped metrics a constant, majority-class predictor can satisfy.
#: ``balanced_accuracy`` is deliberately absent -- a constant predictor scores
#: 0.5 on it, which is the whole reason it is the recommended alternative.
_MAJORITY_SATISFIABLE_METRICS = frozenset({"accuracy", "test_accuracy", "train_accuracy"})


def _majority_class_rate(labels: Any) -> Optional[float]:
    """Share of the most common class, i.e. what predicting one class scores.

    Returns None for anything that is not a concrete 1-D label array -- a study
    whose dataset stage was mocked, or a split that carries no labels. Silence
    is the right answer there: the caller then leaves the target alone rather
    than deriving a floor from something that is not a class distribution.
    """
    try:
        values = np.asarray(labels)
    except Exception:  # noqa: BLE001 - anything unarray-able is simply unknown
        return None
    if values.dtype == object or values.ndim != 1 or values.size == 0:
        return None
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def _effective_target_score(
    request: DatasetStageRequest, labels: Any
) -> tuple[Optional[float], Optional[str]]:
    """Raise an accuracy target that a constant predictor would already meet.

    On an imbalanced dataset the majority-class rate *is* an accuracy floor:
    wine_quality is 86.4% class 0, so the default ``target_score=0.85`` is
    reached by a model that predicts class 0 for every instance. Training then
    stops at 20 epochs reporting ``reached_target: True`` with a model that has
    learned nothing, and the failure only surfaces much later -- in CoXAM's
    surrogate fitting, as "the trained AI model predicts only class [0]", whose
    advice ("train the model longer") does not address the cause.

    So the target is lifted a quarter of the way from the floor to 1.0, which
    forces training to actually separate the classes. Untouched when the caller
    asked for a metric a constant predictor cannot satisfy (balanced_accuracy,
    f1, ...), or when the target already clears the floor.
    """
    target = request.target_score
    if target is None:
        return target, None
    metric = str(request.target_metric or "accuracy").strip().lower()
    if metric not in _MAJORITY_SATISFIABLE_METRICS:
        return target, None
    floor = _majority_class_rate(labels)
    if floor is None or target > floor:
        return target, None
    lifted = min(0.99, floor + (1.0 - floor) * 0.25)
    return lifted, (
        f"target_score={target} is at or below the majority-class rate "
        f"({floor:.4f}) for this dataset, which a model predicting one class for "
        f"every instance already reaches. Raised to {lifted:.4f} so training has "
        f"to separate the classes; pass target_metric='balanced_accuracy' to "
        f"score against a chance-level baseline instead."
    )


def _reject_single_class_model(study: xaikitTest, dataset_id: Optional[str] = None) -> None:
    """Fail here if the trained model predicts one class for everything.

    A constant classifier is not an AI worth explaining: every XAI method
    describes it as flat, CoXAM cannot fit surrogates against it at all, and
    every simulated participant scores its one class. Caught at the stage that
    produced it, naming what to change, rather than several stages downstream.
    """
    data = study.data
    model = study.trained_ai_model
    if data is None or model is None:
        return
    split = getattr(data, "split", None)
    if split is None or getattr(split, "X_model", None) is None:
        return

    from src.ai_models.model_api import _labels_and_scores_from_predictions

    try:
        labels, _scores = _labels_and_scores_from_predictions(model.predict(split.X_model))
        predicted = np.unique(np.asarray(labels))
    except Exception:  # noqa: BLE001
        # This is a safety net, not a correctness gate: a model whose output
        # cannot be read as labels here (a stub in a test, an exotic wrapper)
        # is left to the stages that actually depend on its predictions, which
        # report their own errors. The net must never become the failure.
        return
    if predicted.size != 1:
        return

    floor = _majority_class_rate(getattr(split, "y_train", []))
    where = f" for dataset {dataset_id!r}" if dataset_id else ""
    raise ValueError(
        f"The trained AI model{where} predicts only class {predicted.tolist()} for "
        f"every instance, so it has learned nothing to explain and no XAI method or "
        f"cognitive simulation downstream can produce meaningful results."
        + (f" The training data is {floor:.1%} one class, so a constant predictor "
           f"already scores {floor:.4f} on plain accuracy." if floor else "")
        + " Try model_type='xgboost' (which separates the classes on this data), "
        "target_metric='balanced_accuracy' with target_score around 0.65, or a "
        "higher target_score together with a larger max_epochs."
    )


def _published_weights_path(model_type: str, framework: str, dataset_id: str) -> Optional[Path]:
    """The published checkpoint for this (model, agent, dataset), if shipped."""
    from src.ai_models.paths import model_weights_dir

    path = model_weights_dir(model_type, framework) / f"{dataset_id}_model_weights.pth"
    return path if path.is_file() else None


def _load_published_model(
    study: xaikitTest, dataset_id: str, model_type: str, framework: str
) -> Optional[str]:
    """Load the AI the published study used, instead of training a new one.

    A model trained fresh on the raw dataset is not the AI the human
    participants faced. wine_quality is 86.4% one class, so a fresh MLP
    predicts the minority class for ~7% of instances where the corpus AI
    predicts it for ~50% -- and counterfactual simulation, which asks whether a
    proposed feature change flips the prediction, has almost nothing to flip.
    The published checkpoint reproduces the corpus's own predictions on the
    corpus's instances to 94.5%.

    Returns a note describing what happened, or None when the caller should
    train instead. Nothing is loaded unless the checkpoint's input width
    matches the prepared features: the weights are positional, and feeding them
    a different feature set produces confident nonsense rather than an error
    (measured: 50-51% agreement with the corpus, i.e. chance).
    """
    weights = _published_weights_path(model_type, framework, dataset_id)
    if weights is None:
        return None

    data = study.data
    split = getattr(data, "split", None)
    if split is None or getattr(split, "X_model", None) is None:
        return None
    prepared_width = int(np.asarray(split.X_model).shape[1])

    from src.ai_models.model_api import ModelManager, _shape_from_checkpoint

    try:
        checkpoint_width = int(_shape_from_checkpoint(str(weights))["input_dim"])
    except Exception:  # noqa: BLE001 - an unreadable checkpoint just means "train"
        return None
    if checkpoint_width != prepared_width:
        return (
            f"Published weights for {dataset_id!r} expect {checkpoint_width} features "
            f"but the prepared dataset has {prepared_width}, so a fresh model was "
            "trained instead. The weights are positional; loading them against a "
            "different feature set would silently produce meaningless predictions."
        )

    manager = ModelManager()
    model = manager.load_model(dataset_id, model_type, source=framework)

    # The same study state train_AI_model sets, so every later stage cannot
    # tell the difference between a loaded and a trained model.
    study.model_manager = manager
    study.model = model
    study.trained_ai_model = getattr(model, "engine", model)
    study.model_name = model_type
    study.model_source = framework
    study.training_info = {"loaded_from": str(weights), "trained": False}
    study.ai_predictions_by_instance = None
    study.prediction_table_path = None
    study.prediction_table = None
    return (
        f"Loaded the published {framework} AI from {weights.name} instead of "
        "training a fresh one, so the study explains the same model the human "
        "participants faced. Pass use_published_model=false to train instead."
    )


def _should_try_published_model(
    request: DatasetStageRequest, framework: str, dataset_id: Optional[str] = None
) -> bool:
    """Whether to prefer the published AI over training for this request.

    Automatic only for a CoXAM design on one of the datasets CoXAM's published
    study actually ran (``COXAM_CORPUS_FEATURES``). That restriction is the
    whole justification: loading gives *the AI the human participants faced*,
    which is only true for those datasets. Weights are shipped for others too
    (prima_diabetes, german_credit, ...), but those are simply some other
    checkpoint, and a design using one wants its own trained model.

    CoAX is excluded because its corpus path already avoids training entirely.
    An explicit ``use_published_model`` overrides all of this.
    """
    if request.use_published_model is not None:
        return bool(request.use_published_model)
    if framework != "coxam" or dataset_id is None:
        return False

    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        COXAM_CORPUS_FEATURES,
    )

    return dataset_id in COXAM_CORPUS_FEATURES


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

    framework = design_framework(study)
    published_note = (
        _load_published_model(study, study.data.dataset_id, request.model_type, framework)
        if _should_try_published_model(request, framework, study.data.dataset_id)
        else None
    )
    if published_note and study.trained_ai_model is not None:
        _reject_single_class_model(study)
        study.evaluate(split="both")
        dataset_payload["model"] = {
            "model_name": study.model_name,
            "test_accuracy": float(study.test_accuracy()),
            "metrics": frame_records(study.metrics_table().reset_index()),
            "confusion_matrix": frame_records(study.confusion_matrix_table().reset_index()),
            "model_source": "published_weights",
            "model_source_note": published_note,
        }
        return dataset_payload

    target_score, target_note = _effective_target_score(
        request, getattr(getattr(study.data, "split", None), "y_train", [])
    )
    study.train_AI_model(
        model_type=request.model_type,
        target_metric=request.target_metric,
        target_score=target_score,
        max_epochs=request.max_epochs,
        check_every_epochs=request.check_every_epochs,
        batch_size=request.batch_size,
        verbose=False,
    )
    _reject_single_class_model(study)
    study.evaluate(split="both")

    dataset_payload["model"] = {
        "model_name": study.model_name,
        "test_accuracy": float(study.test_accuracy()),
        "training_summary": frame_records(study.training_summary_table()),
        "training_history": frame_records(study.training_history_table()),
        "metrics": frame_records(study.metrics_table().reset_index()),
        "confusion_matrix": frame_records(study.confusion_matrix_table().reset_index()),
        "target_score_note": target_note,
        "model_source": "trained",
        # Set when published weights existed but could not be used.
        "model_source_note": published_note,
    }
    return dataset_payload


def _finish_multi_dataset_stage(
    study: xaikitTest,
    request: DatasetStageRequest,
    dataset_payload: dict[str, Any],
    dataset_ids: list[str],
    framework: str,
) -> dict[str, Any]:
    """Per dataset, train a real AI model where the published corpus can't
    fully serve it, and skip training where it can -- decided dynamically per
    dataset (``_corpus_skip_reason``/``_coax_needs_real_generation``), not by
    an all-or-nothing rule for the whole study. A multi-dataset design can
    freely mix corpus-covered datasets (e.g. CoAX's adult/wine_quality/
    forest_cover) with one the corpus never collected (e.g. mushrooms), or
    with one whose declared ``xai_method`` levels outgrow what the corpus has
    for it -- each is judged on its own.
    """
    models_payload: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        skip_reason = _corpus_skip_reason(framework, dataset_id)
        needs_real = (
            skip_reason is None
            or _coax_needs_real_generation(study.design_export, framework, dataset_id)
            or _coxam_needs_real_generation(study.design_export, framework)
        )
        if not needs_real:
            # run_coxam_study(source="assets")/run_coax_study(source="corpus")
            # look the corpus's own AI predictions up by model name -- train_AI_model
            # would normally set this (see _finish_dataset_stage), so a
            # corpus-covered dataset that never trains needs it set explicitly,
            # per dataset, or use_dataset's swap restores None for it at
            # simulation time instead of the corpus's own model name.
            study.model_name_by_dataset[dataset_id] = request.model_type
            models_payload[dataset_id] = {"model": None, "model_skipped_reason": skip_reason}
            continue

        published_note = None
        with study.use_dataset(dataset_id):
            if _should_try_published_model(request, framework, dataset_id):
                published_note = _load_published_model(
                    study, dataset_id, request.model_type, framework
                )
            if published_note and study.trained_ai_model is not None:
                _reject_single_class_model(study, dataset_id)
                study.evaluate(split="both")
                models_payload[dataset_id] = {
                    "model": {
                        "model_name": study.model_name,
                        "test_accuracy": float(study.test_accuracy()),
                        "metrics": frame_records(study.metrics_table().reset_index()),
                        "confusion_matrix": frame_records(
                            study.confusion_matrix_table().reset_index()
                        ),
                        "model_source": "published_weights",
                        "model_source_note": published_note,
                    },
                    "model_skipped_reason": None,
                }
                # use_dataset restores the outer state on exit, so the loaded
                # model has to be recorded per dataset -- the same five slots
                # train_AI_model_for_dataset fills.
                study.trained_ai_model_by_dataset[dataset_id] = study.trained_ai_model
                study.model_manager_by_dataset[dataset_id] = study.model_manager
                study.model_by_dataset[dataset_id] = study.model
                study.model_name_by_dataset[dataset_id] = study.model_name
                study.training_info_by_dataset[dataset_id] = study.training_info
                continue
            target_score, target_note = _effective_target_score(
                request, getattr(getattr(study.data, "split", None), "y_train", [])
            )
        study.train_AI_model_for_dataset(
            dataset_id,
            model_type=request.model_type,
            target_metric=request.target_metric,
            target_score=target_score,
            max_epochs=request.max_epochs,
            check_every_epochs=request.check_every_epochs,
            batch_size=request.batch_size,
            verbose=False,
        )
        with study.use_dataset(dataset_id):
            _reject_single_class_model(study, dataset_id)
            study.evaluate(split="both")
            models_payload[dataset_id] = {
                "model": {
                    "model_name": study.model_name,
                    "test_accuracy": float(study.test_accuracy()),
                    "training_summary": frame_records(study.training_summary_table()),
                    "training_history": frame_records(study.training_history_table()),
                    "metrics": frame_records(study.metrics_table().reset_index()),
                    "confusion_matrix": frame_records(study.confusion_matrix_table().reset_index()),
                    "target_score_note": target_note,
                    "model_source": "trained",
                    "model_source_note": published_note,
                },
                "model_skipped_reason": None,
            }
    dataset_payload["models"] = models_payload
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
      ``source="fit"`` can serve it -- and so does one the corpus does cover
      if the design's DVs/``user_task`` also ask for the counterfactual task
      (see ``_coxam_needs_real_generation``): that task evaluates arbitrary
      perturbed instances against the real model, which no static corpus
      table can serve, so skipping training here would only defer the
      failure to a later ``/simulate`` call with a much less clear error.
    * **A multi-dataset study**, per dataset -- ``_finish_multi_dataset_stage``
      trains a real model (via ``train_AI_model_for_dataset``, which trains
      against ``data_by_dataset[dataset_id]`` and stores the result under
      that id rather than on the shared singular fields) for whichever
      dataset(s) the published corpus cannot fully serve, and skips the rest,
      decided independently per dataset -- so a design can freely mix e.g.
      CoAX's corpus-covered adult/wine_quality/forest_cover with mushrooms,
      which the corpus never collected.

    None of that skipping applies when the design also declares
    ``mlProxyBaselines`` (``design.baseline_model_ids``): those run through
    the generic executor in ``run_simulation_stage``, which reads
    ``study.combined_explanations``. Nothing loads a published corpus into
    that table today, so the explanation stage has to *generate* one, and
    generating needs a trained model. A baseline is a virtual participant,
    not a replacement for the explained AI, so in principle it could read the
    corpus's own predictions/explanations and skip training entirely -- see
    the note in ``run_explanations_stage`` for what blocks that.
    """
    design = study.design_export
    dataset_ids = _resolve_dataset_ids(request, design)
    if not dataset_ids:
        raise ValueError(
            "No dataset_id given and the design export does not name one."
        )

    framework = design_framework(study)
    needs_baselines = bool(getattr(design, "baseline_model_ids", None))

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
        # A declared baseline only forces real training when it can't be
        # served from the corpus instead: a baseline (KNN, decision tree, ...)
        # is a proxy that reads predictions/explanations/features -- it does
        # not explain the AI model itself, so a corpus-covered CoAX dataset
        # can serve it exactly like it already serves the primary agent's
        # source="corpus". See _coax_baseline_servable_from_corpus and
        # run_explanations_stage, which loads the corpus into
        # combined_explanations for this case instead of generating a
        # redundant real table. CoXAM's corpus stores fitted DT/LR surrogate
        # models rather than per-instance explanation vectors, so it cannot
        # be served the same way without a surrogate evaluator that does not
        # exist yet -- a CoXAM baseline still forces real training/generation.
        # A CoXAM design that also needs the counterfactual task forces it too
        # (see _coxam_needs_real_generation): the corpus has no live model to
        # evaluate arbitrary perturbed instances against, unlike forward's
        # surrogate-reading path.
        needs_real_ai = (
            _coax_needs_real_generation(design, framework, data.dataset_id)
            or _coxam_needs_real_generation(design, framework)
            or (
                needs_baselines
                and not _coax_baseline_servable_from_corpus(design, framework, data.dataset_id)
            )
        )
        skip_reason = None if needs_real_ai else _corpus_skip_reason(framework, data.dataset_id)
        return _finish_dataset_stage(study, request, dataset_payload, skip_reason)

    # -- multi-dataset, between-subjects ------------------------------------
    if framework == "sim2real":
        raise ValueError(
            f"Sim2Real does not support a multi-dataset study; got {dataset_ids!r}."
        )
    if needs_baselines:
        # An ML-proxy baseline runs through the generic executor, which needs a
        # real trained_ai_model and an explanation table -- and only one of each
        # can exist per study, against one study.data. Failing here names the
        # cause; letting it through would instead skip training (below) and
        # surface as a confusing crash two stages later.
        raise ValueError(
            "mlProxyBaselines are not supported alongside a multi-dataset study: "
            "each baseline needs its own trained AI model and explanation table "
            f"per dataset, and only one of each exists per study. Got datasets "
            f"{dataset_ids!r} with baselines "
            f"{list(getattr(design, 'baseline_model_ids', None) or [])!r}. Drop the "
            "baselines, or run each dataset as its own single-dataset study."
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
    return _finish_multi_dataset_stage(study, request, dataset_payload, dataset_ids, framework)


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

    apparatus_test_ids = sorted(set(getattr(design, "apparatus_instance_ids", [])))
    split_overridden = bool(apparatus_test_ids)
    data_by_dataset = getattr(study, "data_by_dataset", None)

    balance_by_ai_prediction = request.balance_by_ai_prediction
    if balance_by_ai_prediction is None:
        # A model exists only if the dataset stage trained one. Excluded
        # outright, even when a model did get trained (e.g. for a design's
        # own mlProxyBaselines, see run_dataset_stage):
        #
        # * Sim2Real always -- its apparatus instances are a small, fixed,
        #   curated split matching the published corpus, never meant to be
        #   filtered further by a freshly-trained baseline model's predictions.
        # * Any design with a real apparatus-declared instance set
        #   (split_overridden) -- the same reasoning generalized: a small,
        #   researcher-curated instance range (e.g. 20 ids) has no guarantee
        #   of containing both AI-predicted classes, especially for an
        #   imbalanced target, so requiring balance from it is a hard failure
        #   waiting to happen rather than something the range was chosen for.
        balance_by_ai_prediction = (
            design_framework(study) != "sim2real"
            and not split_overridden
            and getattr(study, "trained_ai_model", None) is not None
        )
    #: Datasets that actually got a real split override below (test/train ids
    #: set directly on data.split) -- everything else falls back to the
    #: published corpus's own ids further down, when the framework has one.
    overridden_dataset_ids: set[str] = set()
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
            overridden_dataset_ids.add(dataset_id)
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
        overridden_dataset_ids.add(study.data.dataset_id)

    # Any dataset with no real split override (no apparatus instance ids at
    # all, or a multi-dataset study where only some datasets got one) still
    # needs constraining to whatever its own published corpus can serve, or
    # trials sample freely from the raw dataset's own random split and later
    # fail at simulation time on the first instance the corpus never shipped.
    #
    # Multi-dataset: filtered in place, per dataset, directly on its own
    # split -- never through one flat allowed_instance_ids list. Instance ids
    # are dataset-local (each dataset numbers its own instances from 0), so a
    # flat union across datasets of different corpus sizes would silently
    # admit ids that are only valid for a *different*, larger dataset's range
    # (e.g. wine_quality's real corpus is ids 0-121, but adult's is 0-299 --
    # unioning them would let ids 122-299 leak into wine_quality's pool).
    framework = design_framework(study)
    if isinstance(data_by_dataset, dict) and data_by_dataset:
        for dataset_id, data in data_by_dataset.items():
            if dataset_id in overridden_dataset_ids:
                continue
            corpus_ids = _corpus_instance_ids(framework, dataset_id)
            if not corpus_ids:
                continue
            corpus_set = set(corpus_ids)
            data.split.train_instance_ids = np.asarray(
                sorted(i for i in data.split.train_instance_ids.tolist() if i in corpus_set)
            )
            data.split.test_instance_ids = np.asarray(
                sorted(i for i in data.split.test_instance_ids.tolist() if i in corpus_set)
            )

    allowed_instance_ids = request.allowed_instance_ids
    if allowed_instance_ids is None:
        combined_ids: set[int] = set()
        if not split_overridden:
            # Union, not just the testing set: generate_trials takes one pool
            # for both phases, and a training instance the apparatus declared
            # is still one a human participant was shown. Skipped when the
            # split was already overridden above: that already set the
            # train/test pools to exactly the right ids (including a
            # randomly-sampled training id outside the apparatus's declared
            # set), and this filter would just discard that sample again.
            combined_ids |= set(getattr(design, "apparatus_instance_ids", []))
            combined_ids |= set(getattr(design, "apparatus_training_instance_ids", []))

        # Single-dataset: safe as a flat list here -- there is only one
        # dataset's id space in play, so no cross-dataset leakage is possible.
        if not (isinstance(data_by_dataset, dict) and data_by_dataset):
            if getattr(study, "data", None) is not None and study.data.dataset_id not in overridden_dataset_ids:
                corpus_ids = _corpus_instance_ids(framework, study.data.dataset_id)
                if corpus_ids:
                    combined_ids |= set(corpus_ids)

        allowed_instance_ids = sorted(combined_ids) or None

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

    A declared ``mlProxyBaselines`` model on a CoAX dataset the corpus *does*
    cover is also served without training: a baseline (KNN, decision tree,
    ...) reads predictions/explanations/features, not the explained AI model
    itself, so ``combined_explanations_from_corpus`` reshapes the same corpus
    data ``run_coax_study(source='corpus')`` already reads for the primary
    agent into the shape the generic executor's baseline path
    (``get_trial_instance_explanation``) expects, in place of generating a
    real table. The one thing that used to block this: CoAX's
    ``attribution.csv``/``importance.csv`` carry the *same* ``expMethod`` for
    a given dataset (what separates them is which file, not a column), and
    that lookup keyed on ``expMethod`` alone -- ``combined_explanations_from_corpus``
    tags each row with its own ``xai_type``, and ``get_trial_instance_explanation``
    now disambiguates on it when present. CoXAM's corpus stores fitted DT/LR
    surrogate models rather than per-instance vectors, so it still needs real
    training/generation when a baseline is declared -- no evaluator exists yet
    to read a surrogate's stored structure/coefficients per trial instance.
    """
    design = study.design_export
    framework = design_framework(study)
    needs_baselines = bool(getattr(design, "baseline_model_ids", None))
    trained_by_dataset = _as_dict(getattr(study, "trained_ai_model_by_dataset", None))
    has_a_trained_model = (
        getattr(study, "trained_ai_model", None) is not None
        or any(model is not None for model in trained_by_dataset.values())
    )
    data_by_dataset = _as_dict(getattr(study, "data_by_dataset", None))
    if (
        framework == "coax"
        and needs_baselines
        and not has_a_trained_model
        and not data_by_dataset
        and getattr(study, "data", None) is not None
    ):
        dataset_id = study.data.dataset_id
        pool = study.load_coax_corpus_explanations(dataset_id)
        counts = pool["expMethod"].value_counts().rename_axis("expMethod").reset_index(name="rows")
        return {
            "combined_table": None,
            "rows": int(len(pool)),
            "by_method": frame_records(counts),
            "methods": sorted(
                method for method in pool["expMethod"].astype(str).unique()
                if method != "__prediction_only__"
            ),
            "files": [],
            "skipped_reason": (
                f"Loaded {dataset_id!r}'s published CoAX corpus directly into "
                "combined_explanations for the declared baseline(s) to read -- "
                "no AI model was trained, since a baseline reads predictions/"
                "explanations/features rather than explaining the AI itself."
            ),
        }

    if not needs_baselines and (
        framework in {"coxam", "sim2real"} or (
            framework == "coax" and not has_a_trained_model
        )
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

    if _as_dict(getattr(study, "data_by_dataset", None)):
        # Multi-dataset: only the level(s) run_dataset_stage actually trained
        # (trained_by_dataset) need a real explanation table -- a
        # corpus-covered level has none and needs none, exactly like the
        # single-dataset skip above. Looped rather than one combined table,
        # since expMethod alone cannot disambiguate two datasets' rows and
        # each dataset's own trained model needs its own table anyway.
        by_dataset: dict[str, Any] = {}
        for dataset_id, model in trained_by_dataset.items():
            if model is None:
                by_dataset[dataset_id] = {
                    "combined_table": None, "rows": 0, "by_method": [], "methods": [],
                    "files": [], "skipped_reason": "Served by the published corpus.",
                }
                continue
            path, pool = study.explanations_for_dataset(
                dataset_id,
                methods=request.methods,
                output_dir=f"{request.output_dir}/{dataset_id}",
                target=request.target,
                method_kwargs=request.method_kwargs,
                show_checks=True,
            )
            counts = pool["expMethod"].value_counts().rename_axis("expMethod").reset_index(name="rows")
            by_dataset[dataset_id] = {
                "combined_table": str(path),
                "rows": int(len(pool)),
                "by_method": frame_records(counts),
                "methods": sorted(
                    method for method in pool["expMethod"].astype(str).unique()
                    if method != "__prediction_only__"
                ),
                "files": [str(item) for item in study.explanation_paths_by_dataset.get(dataset_id, [])],
                "skipped_reason": None,
            }
        return {"by_dataset": by_dataset}

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


def _broadcast_coax_params(resolved: Optional[dict[str, Any]]) -> Optional[dict[str, dict[str, Any]]]:
    """One resolved param set, applied to every CoAX condition.

    ``coax_params_for_strategy`` (inside ``coax_models_for_trials``) drops
    whichever keys that condition's actual strategy class does not read, so
    e.g. ``scaling_factor`` only reaches AttributionSum.
    """
    if not resolved:
        return None
    return {xai_type: resolved for xai_type in COAX_STRATEGIES_BY_XAI_TYPE}


def _design_coax_params(study: xaikitTest) -> Optional[dict[str, dict[str, Any]]]:
    """The design export's flat cognitiveConfig, broadcast to every CoAX condition.

    ``request.coax_params`` is keyed by condition (``xai_type`` or
    ``(xai_type, tested_w_xai)``), because that is what a notebook/API caller
    who wants per-condition control needs. The design-planner UI has no such
    per-condition control -- it edits one flat parameter set (retrieval
    threshold, exemplar distance sensitivity, attended features, feature-class
    sensitivity) that is meant to apply everywhere -- so without this, a
    design's User Model page values are silently dropped and every run falls
    back to the fitted population defaults, no matter what the UI shows.
    """
    design = getattr(study, "design_export", None)
    resolved = normalize_cognitive_params("coax", getattr(design, "cognitive_config", None))
    return _broadcast_coax_params(resolved)


def _merge_coxam_task_results(
    study: xaikitTest, coxam_task: Optional[str], results: pd.DataFrame
) -> pd.DataFrame:
    """Fold one CoXAM task's run into the study's running multi-task table.

    CoXAM's forward and counterfactual tasks are separate agents run via
    separate ``/simulate`` calls (``SimulationRequest.coxam_task``), but
    ``study.run_experiment`` overwrites ``study.simulated_results`` on every
    call -- so without this, calling the counterfactual leg after the forward
    one would silently discard the forward rows, and ``posthoc_for``/
    ``analysis_for`` for whichever DV ran first would then 400 with "missing
    columns" the moment the second call finished, trading one unreachable DV
    for the other instead of making both reachable.

    Keyed by the *requested* task, not the resolved one, and only merged
    across calls that explicitly named a task: a caller who never sets
    ``coxam_task`` (the common case, and every call before this field
    existed) keeps today's one-slot, last-call-wins behavior exactly, since
    every such call collapses onto the same ``None`` key and just replaces
    it, matching ``study.run_experiment``'s own overwrite semantics.
    """
    if coxam_task is None:
        study._coxam_task_results = None
        return results
    task_results = getattr(study, "_coxam_task_results", None) or {}
    task_results[coxam_task] = results
    study._coxam_task_results = task_results
    if len(task_results) == 1:
        return results
    merged = pd.concat(list(task_results.values()), ignore_index=True)
    study.simulated_results = merged
    return merged


def _resolve_coax_params(
    study: xaikitTest, request: SimulationRequest
) -> Optional[dict[str, dict[str, Any]]]:
    """``request.coax_params`` in whichever shape the caller actually sent.

    A notebook/API caller wanting per-condition control sends it pre-keyed by
    ``xai_type`` (values are dicts of strategy kwargs). The design-planner UI
    has no per-condition control and instead sends its flat cognitiveConfig
    shape straight through -- ``{"Retrieval Threshold": "-1", ...}``, labels
    mapped to string values -- since that is the only shape it has. Detected
    by whether the top-level values are themselves dicts; a flat shape goes
    through the same label-resolution/type-coercion/broadcast as
    ``_design_coax_params`` instead of being handed to ``coax_models_for_trials``
    unresolved, or rejected as invalid before this code even runs.
    """
    params = request.coax_params
    if params is None:
        return _design_coax_params(study)
    if params and not all(isinstance(value, dict) for value in params.values()):
        return _broadcast_coax_params(normalize_cognitive_params("coax", params))
    return params


def _per_dataset_or_single_source(
    study: xaikitTest, *, trained_value: str, corpus_value: str
) -> str | dict[str, str]:
    """Which runner ``source`` each dataset actually needs.

    A single-dataset study returns one value, exactly as before. A
    multi-dataset study returns a ``{dataset_id: value}`` mapping instead,
    since ``run_dataset_stage`` can train some datasets for real while others
    stay corpus-covered (see ``_finish_multi_dataset_stage``) -- one flat
    value would force every dataset onto whichever source the *first* one
    happened to need. ``xaikitTest._run_multi_dataset_experiment`` resolves a
    dict per level; a plain string is used for every level unchanged.
    """
    data_by_dataset = _as_dict(getattr(study, "data_by_dataset", None))
    if not data_by_dataset:
        return trained_value if getattr(study, "trained_ai_model", None) is not None else corpus_value
    trained_by_dataset = _as_dict(getattr(study, "trained_ai_model_by_dataset", None))
    return {
        level: (trained_value if trained_by_dataset.get(level) is not None else corpus_value)
        for level in data_by_dataset
    }


def _sampling_kwargs(request: SimulationRequest) -> dict[str, Any]:
    """The diverse-mode sampling options, or nothing for every other mode.

    Passed only when the mode asks for them: the runners accept them always,
    but sending them for a shared-parameter run would imply the run sampled
    something when it did not.
    """
    if not is_diverse_mode(request.mode):
        return {}
    return {
        "sampling_seed": request.sampling_seed,
        "sampling_replace": request.sampling_replace,
    }


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
    (with ``condition_filter``), ``whole_experiment`` and
    ``diverse_participant``.

    ``diverse_participant`` runs the whole experiment with one fitted human's
    parameters per virtual participant. Two consequences here. The declared
    ``mlProxyBaselines`` still run in ``whole_experiment`` -- they have no
    fitted population, and ``run_experiment`` refuses the mode for them -- which
    the payload reports as ``baseline_mode`` so the difference is visible rather
    than assumed. And the assignment is saved next to the results as
    ``participant_parameters.csv``, reported under ``files``.

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
            # rather than assuming every design trained a model. A
            # multi-dataset study can mix corpus-covered and trained-for-real
            # datasets, so this is a {dataset_id: source} mapping in that
            # case -- _run_multi_dataset_experiment resolves it per level.
            coax_source = _per_dataset_or_single_source(
                study, trained_value="study", corpus_value="corpus"
            )
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            cognitive_model=coax_models_for_trials(
                pd.DataFrame(study.trials),
                strategies=request.coax_strategies,
                params=_resolve_coax_params(study, request),
            ),
            source=coax_source,
            **_sampling_kwargs(request),
        )
    elif runner == "coxam":
        coxam_source = request.coxam_source
        if coxam_source is None:
            # 'fit' needs trained_ai_model; the dataset stage only trains one
            # when the corpus does not already cover this dataset (see
            # run_dataset_stage), so this follows what actually happened
            # rather than assuming every design trained a model. Same
            # per-dataset mapping as coax_source above for a multi-dataset study.
            coxam_source = _per_dataset_or_single_source(
                study, trained_value="fit", corpus_value="assets"
            )
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            source=coxam_source,
            policy_override=request.coxam_policy,
            eval_params=normalize_cognitive_params("coxam", request.coxam_eval_params),
            user_task=request.coxam_task,
            **_sampling_kwargs(request),
        )
        results = _merge_coxam_task_results(study, request.coxam_task, results)
    elif runner == "sim2real":
        # Counterfactual-only, and its corpus supplies its own stimuli and
        # ground truth, so it needs neither a task switch nor a trained model.
        # Dispatched through run_experiment for the same reason as coax/coxam
        # above -- the per-dataset loop lives there, and store=True is already
        # hardcoded on the far side.
        results = study.run_experiment(
            mode=request.mode,
            participant_id=request.participant_id,
            condition_filter=request.condition_filter,
            strategy=request.sim2real_strategy,
            cognitive_params=normalize_cognitive_params("sim2real", request.sim2real_params),
            normalize_by_i_max=request.sim2real_normalize_by_i_max,
            **_sampling_kwargs(request),
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

    # A design can name mlProxyBaselines alongside a real participant runner
    # (e.g. a Sim2Real design that also wants KNN/Decision-Tree/MLP proxy
    # comparisons) -- each is a separate simulation run against the same
    # trials/explanations, tagged and concatenated rather than the last one
    # silently overwriting study.simulated_results. run_dataset_stage and
    # run_explanations_stage already skip nothing when these are declared, so
    # a trained_ai_model and explanation table are guaranteed to exist here.
    design = study.design_export
    baseline_model_ids = list(getattr(design, "baseline_model_ids", None) or [])
    # A proxy baseline has no fitted human population, so run_experiment
    # refuses diverse_participant for it. Running those comparisons in
    # whole_experiment keeps the mode usable for the designs that declare both,
    # and baseline_mode below says which mode they actually ran in.
    baseline_mode = "whole_experiment" if is_diverse_mode(request.mode) else request.mode
    if runner != "baseline" and baseline_model_ids:
        primary_model_id = study.cognitive_model_id
        primary_cognitive_model = study.cognitive_model
        primary_cognitive_params = study.cognitive_params
        # Each baseline run clears this; the primary run's draw is what the
        # payload and the saved file should describe.
        primary_participant_parameters = study.participant_parameters
        tagged = results.copy()
        tagged["cognitive_model_id"] = primary_model_id
        variants = [tagged]
        for baseline_model_id in baseline_model_ids:
            study.set_cognitive_model(
                cognitive_model_id=baseline_model_id,
                model_kwargs=request.baseline_model_kwargs,
            )
            baseline_results = study.run_experiment(
                mode=baseline_mode,
                participant_id=request.participant_id,
                condition_filter=request.condition_filter,
            )
            if baseline_results.empty:
                continue
            baseline_tagged = baseline_results.copy()
            baseline_tagged["cognitive_model_id"] = baseline_model_id
            variants.append(baseline_tagged)
        # Restore the design's primary model, so the study's "current" model
        # (and the payload below) reflects it rather than the last baseline run.
        study.cognitive_model = primary_cognitive_model
        study.cognitive_model_id = primary_model_id
        study.cognitive_params = primary_cognitive_params
        study.participant_parameters = primary_participant_parameters
        results = pd.concat(variants, ignore_index=True)
        study.simulated_results = results

    csv_path, json_path = study.save_results(out_dir=output_subdir)
    testing = results[results["phase"].astype(str) == "testing"] if "phase" in results else results
    return {
        "runner": runner,
        "cognitive_model_id": study.cognitive_model_id,
        "baseline_model_ids": baseline_model_ids,
        "mode": request.mode,
        "baseline_mode": baseline_mode if baseline_model_ids else None,
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
        "participant_parameters": frame_records(study.participant_parameters),
        "files": {
            "csv": csv_path,
            "json": json_path,
            "participant_parameters": study.participant_parameters_path,
        },
    }


# -- results, analysis, plots --------------------------------------------


def require_results(study: xaikitTest) -> pd.DataFrame:
    if study.simulated_results is None:
        raise ValueError("No simulation has been run for this study yet.")
    return study.simulated_results


def resolve_variable_name(study: xaikitTest, requested: str) -> str:
    """Map a UI display label onto the column name the results table uses.

    The design export keeps both spellings for every IV/CV/DV: ``name`` is the
    slug the planner and the results table use (``xai_type``), ``source_label``
    is the free text the UI shows and posts back (``"XAI Type"``). A plot or
    analysis request phrased in labels would otherwise fail with "Response data
    is missing columns", naming columns that were never supposed to exist.

    An unmatched value is returned unchanged, so a genuinely wrong name still
    raises against the real column list rather than being silently rewritten.
    """
    if not requested:
        return requested
    results = study.simulated_results
    if results is not None and requested in results.columns:
        return requested

    design = getattr(study, "design_export", None)
    wanted = str(requested).strip().casefold()
    for group in ("ivs", "cvs", "dvs", "rvs"):
        for entry in getattr(design, group, None) or []:
            label = str(entry.get("source_label", "")).strip().casefold()
            name = entry.get("name")
            if name and label == wanted:
                return str(name)
    return requested


def _resolve_variable_names(
    study: xaikitTest, requested: Optional[Sequence[str]]
) -> Optional[list[str]]:
    if requested is None:
        return None
    return [resolve_variable_name(study, name) for name in requested]


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
    payload = frame_payload(results, limit=limit, offset=offset)
    payload["plot_variables"] = plot_variables(study)
    return payload


#: Result columns a runner writes for its own bookkeeping. They are real data
#: and stay in the table and the CSV, but they are not experimental factors and
#: must not be offered as plot dimensions -- ``columns`` lists every column, so
#: a picker built from it offers these too.
#:
#: ``explanation_type`` is the concrete surrogate CoXAM showed on a given trial
#: (dt/lr/none). Splitting on it answers a different question than the design's
#: ``xai_type``: a Hybrid participant is assigned Hybrid but is shown dt on some
#: trials and lr on others, so Hybrid disappears from the axis, while the
#: Without-XAI half of ``tested_w_xai`` reappears as a spurious "none" bar.
RUNNER_INTERNAL_COLUMNS = frozenset({
    "explanation_type",
    "selected_strategy",
    "cognitive_model_strategy",
    "condition_name",
    "shown_xai_type",
    "withinCondition",
})


def _reject_runner_internal(study: xaikitTest, *names: Optional[str]) -> None:
    """Refuse to plot against a runner's bookkeeping column.

    Drawing it would succeed and produce a chart that reads as an experimental
    result, so this fails loudly and names the factor that was meant instead.
    """
    offenders = [name for name in names if name in RUNNER_INTERNAL_COLUMNS]
    if not offenders:
        return
    choices = plot_variables(study) or sorted(
        getattr(study, "iv_config", None) or {}
    )
    raise ValueError(
        f"{offenders!r} record what a runner did per trial, not a factor the "
        "study manipulated, so plotting against them misreads the design "
        "(a Hybrid participant is shown dt on some trials and lr on others, so "
        "Hybrid vanishes from the axis and the without-XAI trials appear as a "
        f"'none' condition). Use one of the design's own factors: {choices}."
    )


def plot_variables(study: xaikitTest) -> list[str]:
    """The design's own factors, as the only sane x/split choices for a plot.

    Built from the design export rather than the results columns so a picker
    offers the factors the study actually manipulated -- and nothing a runner
    happened to record alongside them.
    """
    results = study.simulated_results
    if results is None:
        return []
    design = getattr(study, "design_export", None)
    names = [iv["name"] for iv in getattr(design, "ivs", None) or []]
    if not names:
        names = [
            column for column in results.columns
            if column not in RUNNER_INTERNAL_COLUMNS
        ]
    return [name for name in names if name in results.columns]


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
    ivs = (
        _resolve_variable_names(study, ivs)
        if ivs
        else [iv["name"] for iv in getattr(design, "between_ivs", [])]
    )
    dvs = (
        _resolve_variable_names(study, dvs)
        if dvs
        else list(getattr(design, "simulatable_dvs", []))
    )
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
    dv = resolve_variable_name(study, dv)
    condition_cols = (
        _resolve_variable_names(study, condition_cols)
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
    # The UI posts the design's display labels ("XAI Type"), not the slugged
    # column names the results table carries ("xai_type").
    x_iv = resolve_variable_name(study, x_iv)
    hue_iv = resolve_variable_name(study, hue_iv)
    dv = resolve_variable_name(study, dv)
    _reject_runner_internal(study, x_iv, hue_iv)
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
    # Explicit lists come from the UI in display labels; the design's own
    # entries are already slugged.
    ivs = (
        _resolve_variable_names(study, ivs)
        if ivs
        else [iv["name"] for iv in getattr(design, "ivs", [])]
    )
    dvs = (
        _resolve_variable_names(study, dvs)
        if dvs
        else list(getattr(design, "simulatable_dvs", []))
    )
    _reject_runner_internal(study, *ivs)
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


# -- human-vs-model comparison (precomputed, no fitting at request time) --

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"
HUMAN_COMPARISON_STUDIES = ("coax", "coxam", "sim2real")


def human_comparison_payload(study: str) -> Optional[dict[str, Any]]:
    """A precomputed study's comparison PNG and NLL/BIC table, read from disk.

    Everything here was built ahead of time by ``assets/build_human_vs_model_plot_data.py``,
    ``assets/build_human_vs_model_plots.py`` and ``assets/build_human_vs_model_fit_stats.py``
    -- this only reads and re-serves those files. Returns ``None`` if ``study``
    is not one of ``coax``/``coxam``/``sim2real``, or if its files have not
    been built yet.
    """
    import base64
    import json

    key = str(study).lower().strip()
    if key not in HUMAN_COMPARISON_STUDIES:
        return None

    plot_path = ASSETS_ROOT / "human_vs_model_plots" / f"{key}.png"
    panels_path = ASSETS_ROOT / "human_vs_model_plot_data.json"
    fit_stats_path = ASSETS_ROOT / "human_vs_model_fit_stats.json"
    if not plot_path.is_file():
        return None

    payload: dict[str, Any] = {
        "study": key,
        "plot_png": "data:image/png;base64," + base64.b64encode(plot_path.read_bytes()).decode("ascii"),
        "panels": None,
        "fit_stats": None,
    }
    if panels_path.is_file():
        panel_data = json.loads(panels_path.read_text())
        payload["panels"] = next(
            (s for s in panel_data.get("studies", []) if s.get("name", "").lower() == key),
            None,
        )
    if fit_stats_path.is_file():
        fit_data = json.loads(fit_stats_path.read_text())
        payload["fit_stats"] = next(
            (s for s in fit_data.get("studies", []) if s.get("name", "").lower() == key),
            None,
        )
    return payload

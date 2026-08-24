"""Request bodies for the study API.

Defaults are deliberately ``None`` wherever the design export already answers
the question -- the pipeline fills those from the export so the UI never has to
restate what it exported.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class CreateStudyRequest(BaseModel):
    """The experiment-design UI export, posted as-is."""

    design: dict[str, Any] = Field(..., description="The experiment-design JSON export.")
    project_name: str = Field("xaikit_study", description="Name used for output files.")


class DatasetStageRequest(BaseModel):
    """Dataset preparation and AI-model training."""

    dataset_id: Optional[Union[str, list[str]]] = Field(
        None,
        description=(
            "Defaults to the export's dataset(s). Pass a list of 2+ ids for a "
            "multi-dataset, between-subjects study -- or declare a between-"
            "subjects IV named 'dataset' with 2+ levels in the design export "
            "itself, which this defaults to when unset."
        ),
    )
    feature_cols: Optional[list[str]] = None
    num_features: Optional[int] = None
    rank_features_by_target: bool = False
    model_type: str = "mlp"
    test_size: float = 0.4
    random_state: int = 42
    target_metric: str = "accuracy"
    target_score: Optional[float] = 0.85
    max_epochs: int = 50
    check_every_epochs: int = 20
    batch_size: int = 512


class TrialsStageRequest(BaseModel):
    """Trial generation. The export gives the totals; the split is stated here."""

    participants_per_between_condition: Optional[int] = Field(
        None, description="Defaults to the export's participants per condition."
    )
    num_training: Optional[int] = Field(
        None,
        description=(
            "Defaults to trials_per_participant minus the apparatus's declared "
            "testing instance count when the design has one (so a design with "
            "30 trials/participant and 20 apparatus testing instances defaults "
            "to 10, not an arbitrary flat number), else 4."
        ),
    )
    num_testing: Optional[int] = Field(
        None,
        description=(
            "Defaults to the export's trials per participant minus num_training."
        ),
    )
    allowed_instance_ids: Optional[list[int]] = Field(
        None,
        description=(
            "Restrict trials to these instances. Defaults to the union of "
            "every apparatus[].params.instanceIds / trainingInstanceIds the "
            "design export declared, so simulated trials reference exactly "
            "the instances the human-study UI showed participants -- pass an "
            "explicit list (or [] for no restriction) to override."
        ),
    )
    balance_by_ai_prediction: Optional[bool] = Field(
        None,
        description=(
            "Draw each phase equally from both predicted classes. Requires a "
            "trained AI model, so the default is True when the dataset stage "
            "trained one and False otherwise -- Sim2Real designs skip training "
            "(see the /dataset endpoint), so their default is False. Pass "
            "True explicitly to require balancing and get a clear error if no "
            "model was trained, rather than picking a value silently."
        ),
    )
    seed: int = 42
    output_dir: str = "trials"
    preview_rows: int = 20


class ExplanationStageRequest(BaseModel):
    """XAI generation. Methods come from the design unless overridden."""

    methods: Optional[list[str]] = None
    target: int = 1
    method_kwargs: Optional[dict[str, dict[str, Any]]] = Field(
        default_factory=lambda: {"lime": {"num_samples": 1000}}
    )
    output_dir: str = "generated_explanation"


class SimulationRequest(BaseModel):
    """One virtual-participant run.

    ``mode`` is the API layer's selection vocabulary, so a single-trial
    walkthrough and the full experiment use the same endpoint.

    ``coax_strategies``/``coax_params`` are keyed by ``xai_type``. Leaving them
    unset uses the library defaults (SensitiveFeatures / ImportanceCategorization
    / AttributionSum at sensitivity 20, k 3, retrieval_threshold -2.3), which are
    *not* the hand-tuned values in the tutorial notebooks -- pass them here to
    reproduce a notebook run exactly.

    ``coax_params`` also accepts the design-planner UI's flat cognitiveConfig
    shape -- ``{"Retrieval Threshold": "-1", ...}``, labels mapped to string
    values rather than xai_type mapped to kwarg dicts -- since that is the
    only shape the UI has to send; ``dict[str, Any]`` (rather than
    ``dict[str, dict[str, Any]]``) lets that through so the pipeline can
    resolve it instead of pydantic rejecting each label's string value as
    "not a valid dictionary". See ``pipeline._resolve_coax_params``.
    """

    mode: str = Field(
        "whole_experiment",
        description=(
            "trial_by_trial | participant_by_participant | whole_condition | "
            "whole_experiment | diverse_participant. The last runs the whole "
            "experiment but gives every virtual participant its own cognitive "
            "parameters, drawn from the humans the selected agent was fitted to "
            "and filtered to that participant's own condition. Research agents "
            "only -- a baseline model has no fitted population and is refused."
        ),
    )
    participant_id: Optional[int] = None
    condition_filter: Optional[dict[str, Any]] = None
    sampling_seed: int = Field(
        0,
        description=(
            "diverse_participant only. Makes the participant -> fitted-human "
            "assignment reproducible across runs."
        ),
    )
    sampling_replace: Optional[bool] = Field(
        None,
        description=(
            "diverse_participant only. None (the default) deals distinct fitted "
            "people while the pool lasts, then reshuffles; true draws i.i.d. "
            "instead, which lets one person be dealt to several participants."
        ),
    )
    coax_strategies: Optional[dict[str, str]] = None
    coax_params: Optional[dict[str, Any]] = None
    coax_source: Optional[str] = Field(
        None,
        description=(
            "CoAX designs only. 'study' scores against this study's own trained "
            "model and explanation table; 'corpus' reads the published CoAX "
            "user-study corpus instead, needing neither. Defaults to 'study' "
            "when the dataset stage trained a model and 'corpus' when it didn't "
            "(a corpus-covered dataset -- see /dataset's model_skipped_reason)."
        ),
    )
    coxam_task: Optional[str] = Field(
        None,
        description=(
            "CoXAM designs only. 'forward' or 'counterfactual', forcing which "
            "agent this call runs. CoXAM runs the two tasks as separate agents "
            "(see DesignExport.user_tasks), so a design naming both "
            "forward_accuracy and counterfactual_accuracy as DVs needs one "
            "/simulate call per task -- pass this to get the second one, then "
            "combine the two runs' results. Defaults to whatever the trials' own "
            "user_task column says (from a design that varies task as an IV), "
            "or forward_accuracy if neither says otherwise."
        ),
    )
    coxam_policy: Optional[str] = Field(
        None,
        description=(
            "CoXAM designs only. One of 'lr_calculation' | 'lr_heuristic' | "
            "'dt_traversal' to force that one sub-strategy for every trial "
            "instead of letting the trained meta-policy choose. Omit to use the "
            "meta-policy."
        ),
    )
    coxam_eval_params: Optional[dict[str, float]] = Field(
        None,
        description=(
            "CoXAM designs only. Fixed values for the three parameters that are "
            "free at evaluation time -- decision_noise [0.3, 0.7], "
            "memory_recall_threshold [-1.0, 2.0] and opportunity_cost "
            "[0.0, 0.02]. Omit to let the environment sample them."
        ),
    )
    coxam_source: Optional[str] = Field(
        None,
        description=(
            "CoXAM designs only. 'fit' fits decision-tree and logistic-regression "
            "surrogates against this study's own trained model; 'assets' loads the "
            "pre-generated tables from assets/explanations/CoXAM instead, which "
            "requires trials constrained to that corpus' instance ids. Defaults to "
            "'assets' when the dataset stage skipped training (a corpus-covered "
            "dataset -- see /dataset's model_skipped_reason) and 'fit' otherwise, "
            "since 'fit' would simply fail with no trained_ai_model to fit against."
        ),
    )
    sim2real_strategy: str = Field(
        "auto",
        description=(
            "Sim2Real designs only. Which cognitive model simulates a participant. "
            "'auto' (the default) picks per condition: 'sparse' runs "
            "'sensitive_features' because its explanation almost never shows the "
            "changed feature (6.9% of trials), leaving 'attribution_sum' nothing to "
            "read -- it degenerates to a constant response scoring 0.310 against a "
            "real human 0.757, where exemplar similarity scores 0.762. The other "
            "three conditions keep 'attribution_sum', which reads the attributions "
            "they do carry. Naming a strategy explicitly ('attribution_sum', "
            "'sensitive_features', 'salient_features') forces it on every condition "
            "instead -- note the exemplar strategies ignore the explanation, so "
            "forcing one returns the same accuracy for all four conditions."
        ),
    )
    sim2real_params: Optional[dict] = Field(
        None,
        description=(
            "Sim2Real designs only. Overrides for the cognitive parameters, applied "
            "on top of the per-condition fitted population. Anything omitted keeps "
            "its fitted value, so passing one field does not reset the rest."
        ),
    )
    sim2real_normalize_by_i_max: bool = Field(
        False,
        description=(
            "Sim2Real designs only. Rescale each condition's attributions by its "
            "i_max before the model reads them. FITTED_SIM2REAL_PARAMS was fitted "
            "with this off, and measured simulated forward accuracy against real "
            "human accuracy confirms it: turning it on leaves three of four "
            "conditions unchanged but pushes 'faithful' from a close match "
            "(-0.039) to a much larger miss (+0.145)."
        ),
    )
    baseline_model_id: Optional[str] = Field(
        None,
        description=(
            "Non-CoAX designs only, e.g. 'knn'. Overrides the design's own "
            "userModel selection; omit it to run whatever the design chose."
        ),
    )
    baseline_model_kwargs: Optional[dict[str, Any]] = None
    preview_rows: int = 20


class AnalysisRequest(BaseModel):
    """IV x DV analysis. Empty lists mean 'whatever the design declared'."""

    ivs: Optional[list[str]] = None
    dvs: Optional[list[str]] = None


class PostHocRequest(BaseModel):
    """Pairwise condition comparisons for one DV, with corrected p-values."""

    dv: str
    condition_cols: Optional[list[str]] = Field(
        None, description="Defaults to every IV the design declares."
    )
    correction: Optional[str] = Field(
        "holm", description="Multiple-comparison correction; pass null for raw p-values."
    )
    phase: str = "testing"


class InteractionPlotRequest(BaseModel):
    """One DV against two IVs, returned as the aggregate the UI draws."""

    x_iv: str
    hue_iv: str
    dv: str
    phase: Optional[str] = "testing"
    errorbar: Optional[str] = "ci95"
    title: Optional[str] = None
    include_png: bool = False


class GridPlotRequest(BaseModel):
    """Every DV against every IV, returned as one aggregate table."""

    ivs: Optional[list[str]] = None
    dvs: Optional[list[str]] = None
    phase: Optional[str] = "testing"
    errorbar: Optional[str] = "ci95"
    title: Optional[str] = "Experiment results"
    include_png: bool = False

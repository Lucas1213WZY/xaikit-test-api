"""Request bodies for the study API.

Defaults are deliberately ``None`` wherever the design export already answers
the question -- the pipeline fills those from the export so the UI never has to
restate what it exported.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateStudyRequest(BaseModel):
    """The experiment-design UI export, posted as-is."""

    design: dict[str, Any] = Field(..., description="The experiment-design JSON export.")
    project_name: str = Field("xaikit_study", description="Name used for output files.")


class DatasetStageRequest(BaseModel):
    """Dataset preparation and AI-model training."""

    dataset_id: Optional[str] = Field(None, description="Defaults to the export's dataset.")
    feature_cols: Optional[list[str]] = None
    num_features: Optional[int] = None
    rank_features_by_target: bool = False
    model_type: str = "mlp"
    test_size: float = 0.4
    random_state: int = 42
    target_metric: str = "accuracy"
    target_score: Optional[float] = 0.90
    max_epochs: int = 1000
    check_every_epochs: int = 10
    batch_size: int = 100


class TrialsStageRequest(BaseModel):
    """Trial generation. The export gives the totals; the split is stated here."""

    participants_per_between_condition: Optional[int] = Field(
        None, description="Defaults to the export's participants per condition."
    )
    num_training: int = 4
    num_testing: Optional[int] = Field(
        None,
        description=(
            "Defaults to the export's trials per participant minus num_training."
        ),
    )
    balance_by_ai_prediction: bool = True
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
    """

    mode: str = Field(
        "whole_experiment",
        description=(
            "trial_by_trial | participant_by_participant | whole_condition | "
            "whole_experiment"
        ),
    )
    participant_id: Optional[int] = None
    condition_filter: Optional[dict[str, Any]] = None
    coax_strategies: Optional[dict[str, str]] = None
    coax_params: Optional[dict[str, dict[str, Any]]] = None
    baseline_model_id: Optional[str] = Field(
        None,
        description="Non-CoAX designs only, e.g. 'knn_baseline'.",
    )
    baseline_model_kwargs: Optional[dict[str, Any]] = None
    preview_rows: int = 20


class AnalysisRequest(BaseModel):
    """IV x DV analysis. Empty lists mean 'whatever the design declared'."""

    ivs: Optional[list[str]] = None
    dvs: Optional[list[str]] = None


class InteractionPlotRequest(BaseModel):
    """One DV against two IVs, returned as the aggregate the UI draws."""

    x_iv: str
    hue_iv: str
    dv: str
    phase: Optional[str] = "testing"
    errorbar: Optional[str] = "sem"
    title: Optional[str] = None
    include_png: bool = False


class GridPlotRequest(BaseModel):
    """Every DV against every IV, returned as one aggregate table."""

    ivs: Optional[list[str]] = None
    dvs: Optional[list[str]] = None
    phase: Optional[str] = "testing"
    errorbar: Optional[str] = "sem"
    title: Optional[str] = "Experiment results"
    include_png: bool = False

"""High-level XAIKit workflow API."""

from __future__ import annotations

import io
import base64
import html
import json
import uuid
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.cognitive_models import (
    default_cognitive_params,
    dummy_cognitive_model,
)
from src.data_loaders import PreparedDataset, prepare_dataset, reencode_prepared_dataset
from src.experiment_planner import (
    TrialGenerationResult,
    ValidationReport,
    init_experiment_config,
    select_trial_rows,
    load_support_matrix,
    set_factor,
    set_iv,
    validate_experiment_config,
    validate_xaikit_test,
)
from src.workflow_standard import (
    DEFAULT_EXPLANATION_INSTANCE_LIMIT,
    ensure_prediction_coverage,
)
import src.xai_adapter as xai_adapter_api
import src.cognitive_models as cognitive_models_api
import src.experiment_planner as experiment_planner_api
import src.virtual_experiment_executor as virtual_experiment_api
from src.ai_models import evaluation as ai_eval
from src.experiment_planner import preview as ep_preview
from src.experiment_planner.protocol import (
    default_study_protocol,
    edit_study_protocol,
    normalize_study_protocol,
    validate_study_protocol,
)


class xaikitTest:
    """Collate one XAI experiment workflow into a single reusable object."""

    def __init__(
        self,
        project_name: str = "xaikit_test",
        *,
        output_dir: str | Path = ".",
        auto_validate_design: bool = True,
    ) -> None:
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.auto_validate_design = auto_validate_design

        self.iv_config, self.CVs, self.DVs = init_experiment_config()

        self.data: Optional[PreparedDataset] = None
        self.model_manager = None
        self.model = None
        self.trained_ai_model = None
        self.model_name: Optional[str] = None
        self.model_source: Optional[str] = None
        self.training_info: Optional[dict[str, Any]] = None
        self.training_stdout: str = ""
        self.metrics: dict[str, dict[str, Any]] = {}

        self.trial_config = None
        self.trial_result: Optional[TrialGenerationResult] = None
        self.trials: list[dict[str, Any]] = []
        self.ai_predictions_by_instance: Optional[dict[int, Any]] = None

        self.explanation_config = None
        self.explanation_paths: list[Path] = []
        self.explanation_dfs: list[pd.DataFrame] = []
        self.prediction_table_path: Optional[Path] = None
        self.prediction_table: Optional[pd.DataFrame] = None
        self.combined_explanation_path: Optional[Path] = None
        self.combined_explanations: Optional[pd.DataFrame] = None

        self.cognitive_params: dict[str, float] = default_cognitive_params()
        self.cognitive_model: Callable[..., dict[str, Any]] = dummy_cognitive_model
        self.cognitive_model_id: str = "placeholder"
        self.validation_reports: dict[str, ValidationReport] = {}
        self.simulated_results: Optional[pd.DataFrame] = None
        self.simulated_csv_path: Optional[str] = None
        self.simulated_json_path: Optional[str] = None
        self.study_protocol: dict[str, Any] = default_study_protocol()
        self.walkthrough_previewed: bool = False
        self.walkthrough_approved: bool = False

    def set_study_protocol(
        self,
        *,
        study_title: str,
        research_questions: Sequence[str] | str,
        consent_text: str,
        procedure_steps: Sequence[dict[str, Any]],
        study_summary: str = "",
        start_survey_questions: Sequence[str] | str = (),
        end_survey_questions: Sequence[str] | str = (),
        validate: bool = True,
    ) -> "xaikitTest":
        """Store the researcher-authored, participant-facing study protocol."""
        protocol = normalize_study_protocol({
            "study_title": study_title,
            "research_questions": research_questions,
            "study_summary": study_summary,
            "consent_text": consent_text,
            "start_survey_questions": start_survey_questions,
            "end_survey_questions": end_survey_questions,
            "procedure_steps": list(procedure_steps),
        })
        problems = validate_study_protocol(protocol) if validate else []
        if problems:
            raise ValueError("Study setup is incomplete: " + " ".join(problems))
        self.study_protocol = protocol
        self.walkthrough_previewed = False
        self.walkthrough_approved = False
        return self

    def edit_study_protocol(self) -> Any:
        """Show an interactive notebook form that saves values on this object."""
        def store(protocol: dict[str, Any]) -> None:
            self.study_protocol = normalize_study_protocol(protocol)
            self.walkthrough_previewed = False
            self.walkthrough_approved = False

        return edit_study_protocol(self.study_protocol, on_save=store)

    def save_study_protocol(self, path: str | Path = "study_protocol.json") -> str:
        """Validate and export the study setup to JSON."""
        problems = validate_study_protocol(self.study_protocol)
        if problems:
            raise ValueError("Study setup is incomplete: " + " ".join(problems))
        output_path = self._resolve_output_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.study_protocol, indent=2), encoding="utf-8")
        return str(output_path)

    def approve_walkthrough(self, *, confirmed: bool = False) -> "xaikitTest":
        """Approve a completed walkthrough, with explicit confirmation required."""
        if not confirmed:
            raise ValueError("Pass confirmed=True only after reviewing the complete walkthrough.")
        if not self.walkthrough_previewed:
            raise RuntimeError("Preview the experiment walkthrough before approving it.")
        problems = validate_study_protocol(self.study_protocol)
        if problems:
            raise ValueError("Study setup is incomplete: " + " ".join(problems))
        self.walkthrough_approved = True
        return self

    def guide(self, stage: str = "design") -> Optional[pd.DataFrame]:
        """Print a concise guide for one workflow stage."""
        key = stage.lower().strip().replace("-", "_").replace(" ", "_")
        aliases = {
            "iv": "design",
            "dv": "design",
            "cv": "design",
            "variables": "design",
            "data": "dataset",
            "trials": "trial_generation",
            "trial": "trial_generation",
            "training": "model_training",
            "model": "model_training",
            "xai": "explanation_generation",
            "explanations": "explanation_generation",
            "agents": "cognitive_models",
            "agent": "cognitive_models",
            "cognitive": "cognitive_models",
            "cognitive_agent": "cognitive_models",
            "cognitive_agents": "cognitive_models",
            "cognitive_model": "cognitive_models",
            "cognitive_models": "cognitive_models",
            "execution": "cognitive_simulation",
            "simulation": "cognitive_simulation",
        }
        key = aliases.get(key, key)
        if key not in _GUIDE_MESSAGES:
            available = ", ".join(_GUIDE_MESSAGES)
            raise ValueError(f"Unknown guide stage {stage!r}. Use one of: {available}.")
        if key == "cognitive_models":
            print(_cognitive_model_guide(self))
            return cognitive_model_guide_table()
        print(_GUIDE_MESSAGES[key])
        return None

    def guide_design(self) -> None:
        """Print the experimental-design guide."""
        self.guide("design")

    def guide_dataset(self) -> None:
        """Print the dataset-preparation guide."""
        self.guide("dataset")

    def guide_trial_generation(self) -> None:
        """Print the trial-generation guide."""
        self.guide("trial_generation")

    def guide_model_training(self) -> None:
        """Print the AI-model-training guide."""
        self.guide("model_training")

    def guide_explanation_generation(self) -> None:
        """Print the XAI-generation guide."""
        self.guide("explanation_generation")

    def guide_cognitive_models(self) -> pd.DataFrame:
        """Print the cognitive-model selection guide."""
        return self.guide("cognitive_models")

    def guide_cognitive_simulation(self) -> None:
        """Print the cognitive-simulation guide."""
        self.guide("cognitive_simulation")

    def set_design(
        self,
        *,
        iv_config: Optional[dict[str, dict[str, Any]]] = None,
        cvs: Optional[dict[str, list[Any]]] = None,
        dvs: Optional[dict[str, list[Any]]] = None,
        show: bool = True,
    ) -> "xaikitTest":
        """Replace the stored IV/CV/DV design dictionaries."""
        if iv_config is not None:
            self.iv_config = deepcopy(iv_config)
        if cvs is not None:
            self.CVs = deepcopy(cvs)
        if dvs is not None:
            self.DVs = deepcopy(dvs)
        if self.auto_validate_design:
            self.validate_design(show=show)
        return self

    def add_iv(
        self,
        name: str,
        iv_type: str,
        levels: list[Any],
        *,
        randomization: str = "block",
        show: bool = False,
    ) -> "xaikitTest":
        """Add or replace one independent variable."""
        set_iv(self.iv_config, name, iv_type, levels, randomization=randomization)
        if show:
            self.validate_design(show=True)
        return self

    def add_cv(self, name: str, levels: list[Any], *, show: bool = False) -> "xaikitTest":
        """Add or replace one control variable."""
        set_factor(self.CVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def add_dv(self, name: str, levels: list[Any], *, show: bool = False) -> "xaikitTest":
        """Add or replace one dependent variable."""
        set_factor(self.DVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def validate_design(self, *, show: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Validate and optionally print the current experimental design."""
        return validate_experiment_config(self.iv_config, self.CVs, self.DVs, show=show)

    def validate(
        self,
        *,
        stage: str = "design",
        strict: bool = False,
        show: bool = True,
    ) -> ValidationReport:
        """Validate this workflow object against the XAIKit support standard."""
        report = validate_xaikit_test(
            self,
            stage=stage,
            strict=strict,
            show=show,
        )
        self.validation_reports[report.stage] = report
        return report

    def prepare_dataset(
        self,
        dataset_id: str,
        *,
        model_type: str = "mlp",
        feature_cols: Optional[Sequence[str]] = None,
        num_features: Optional[int] = None,
        rank_features_by_target: bool = True,
        use_default_features: bool = True,
        requires_one_hot_encoding: Optional[bool] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        show_available: bool = True,
        show_summary: bool = True,
    ) -> PreparedDataset:
        """Load, optionally feature-select, and split the dataset."""
        self.data = prepare_dataset(
            dataset_id,
            model_type=model_type,
            feature_cols=feature_cols,
            num_features=num_features,
            rank_features_by_target=rank_features_by_target,
            use_default_features=use_default_features,
            requires_one_hot_encoding=requires_one_hot_encoding,
            test_size=test_size,
            random_state=random_state,
            show_available=show_available,
            show_summary=show_summary,
        )
        return self.data

    def generate_trials(
        self,
        *,
        model_name: Optional[str] = None,
        participants_per_between_condition: int = 24,
        num_training: int = 0,
        num_testing: int = 12,
        balance_by_ai_prediction: bool = False,
        counterbalancing_strategy: str = "auto",
        trial_randomization_strategy: str = "balanced",
        instance_wise_explanation: bool = False,
        shuffle_instances: bool = True,
        max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT,
        seed: int = 42,
        output_dir: str | Path = "experiment_output",
        preview_rows: int = 10,
        show: bool = True,
    ) -> TrialGenerationResult:
        """Build training rows followed by held-out testing rows."""
        data = self._require_data()
        ai_predictions_by_instance = None
        if balance_by_ai_prediction:
            from src.workflow_standard import prediction_labels

            trained_ai_model = self._require_trained_ai_model()
            predictions = prediction_labels(
                trained_ai_model.predict(data.split.X_model)
            )
            ai_predictions_by_instance = {
                int(instance_id): _as_python_scalar(prediction)
                for instance_id, prediction in zip(
                    data.split.raw_instance_ids,
                    predictions,
                )
            }
            self.ai_predictions_by_instance = ai_predictions_by_instance
        if model_name is not None:
            self.model_name = model_name
        self.trial_config = experiment_planner_api.init_trial_build_config(
            data=data,
            iv_config=self.iv_config,
            cvs=self.CVs,
            model_name=model_name,
            participants_per_between_condition=participants_per_between_condition,
            num_training=num_training,
            num_testing=num_testing,
            ai_predictions_by_instance=ai_predictions_by_instance,
            counterbalancing_strategy=counterbalancing_strategy,
            trial_randomization_strategy=trial_randomization_strategy,
            instance_wise_explanation=instance_wise_explanation,
            shuffle_instances=shuffle_instances,
            max_trial_instances=max_trial_instances,
            seed=seed,
            output_dir=self._resolve_output_path(output_dir),
        )
        self.trial_result = experiment_planner_api.generate_experimental_trials(
            self.trial_config,
            show=show,
            preview_rows=preview_rows,
        )
        self.trials = self.trial_result.trials
        return self.trial_result

    def train_AI_model(
        self,
        *,
        model_type: str = "mlp",
        source: Optional[str] = None,
        target_accuracy: Optional[float] = None,
        target_metric: str = "accuracy",
        target_score: Optional[float] = None,
        max_epochs: int = 300,
        check_every_epochs: int = 10,
        batch_size: int = 1000,
        verbose: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create and train the AI model used for predictions and explanations."""
        data = self._require_data()
        from src.ai_models import ModelManager, requires_one_hot_encoding

        required_one_hot = requires_one_hot_encoding(model_type)
        if data.split.one_hot_encode != required_one_hot:
            data = reencode_prepared_dataset(
                data,
                model_type=model_type,
                requires_one_hot_encoding=required_one_hot,
                show_summary=verbose,
            )
            self.data = data

        def _create_and_train() -> None:
            self.model_name = model_type
            self.model_manager = ModelManager()
            self.model = self.model_manager.create_model(
                dataset=data.dataset_id,
                model_type=model_type,
                input_dim=data.X_train.shape[1],
                num_classes=len(set(data.y_train.tolist())),
                source=source,
                **(model_kwargs or {}),
            )

            stop_score = target_score if target_score is not None else target_accuracy
            if stop_score is None:
                self.training_info = self.model_manager.train(
                    data.X_train,
                    data.y_train,
                    batch_size=batch_size,
                    **(train_kwargs or {}),
                )
            else:
                self.training_info = self.model_manager.train_until_accuracy(
                    data.X_train,
                    data.y_train,
                    target_accuracy=stop_score,
                    target_metric=target_metric,
                    max_epochs=max_epochs,
                    check_every_epochs=check_every_epochs,
                    batch_size=batch_size,
                    **(train_kwargs or {}),
                )

        if verbose:
            _create_and_train()
            self.training_stdout = ""
        else:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                _create_and_train()
            self.training_stdout = buffer.getvalue()

        self.trained_ai_model = getattr(self.model, "engine", self.model)
        self.model_source = source
        self.ai_predictions_by_instance = None
        self.prediction_table_path = None
        self.prediction_table = None
        return self.model

    def _dataset_id(self) -> Any:
        return self.data.dataset_id if self.data is not None else None

    def training_summary_table(self) -> pd.DataFrame:
        """Return a one-row summary of the latest training run."""
        return ai_eval.training_summary_table(self.training_info, self.model_name, self._dataset_id())

    def training_history_table(self) -> pd.DataFrame:
        """Return the accuracy checkpoint history from the latest training run."""
        return ai_eval.training_history_table(self.training_info, self.model_name, self._dataset_id())

    def plot_training_history(self, *, ax: Any = None) -> Any:
        """Plot metric checkpoints from training with a target score."""
        return ai_eval.plot_training_history(self.training_info, ax=ax)

    def test_accuracy(self) -> float:
        """Return held-out test accuracy for the trained AI model."""
        data = self._require_data()
        manager = self._require_model_manager()
        return manager.test_accuracy(data.X_test, data.y_test)

    def confusion_matrix_table(
        self,
        *,
        split: str = "test",
        positive_label: int = 1,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Return a labeled confusion matrix for the requested split."""
        return ai_eval.confusion_matrix_table(
            self._require_model_manager(), self._require_data(),
            split=split, positive_label=positive_label, threshold=threshold,
        )

    def plot_confusion_matrix(
        self,
        *,
        split: str = "test",
        positive_label: int = 1,
        threshold: float = 0.5,
        ax: Any = None,
    ) -> Any:
        """Plot a labeled confusion matrix for the requested split."""
        return ai_eval.plot_confusion_matrix(
            self._require_model_manager(), self._require_data(),
            split=split, positive_label=positive_label, threshold=threshold, ax=ax,
        )

    def evaluate(
        self,
        *,
        split: str = "both",
        positive_label: int = 1,
        threshold: float = 0.5,
        include_report: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate the trained AI model with classic classification metrics."""
        results = ai_eval.evaluate_model(
            self._require_model_manager(), self._require_data(),
            split=split, positive_label=positive_label,
            threshold=threshold, include_report=include_report,
        )
        self.metrics.update(results)
        return results

    def metrics_table(self) -> pd.DataFrame:
        """Return scalar evaluation metrics as a compact split-by-metric table."""
        return ai_eval.metrics_table(self.metrics)

    def plot_auc_curves(
        self,
        *,
        split: str = "both",
        positive_label: int = 1,
        ax: Any = None,
    ) -> Any:
        """Plot ROC curves and AUC values for train/test predictions."""
        return ai_eval.plot_auc_curves(
            self._require_model_manager(), self._require_data(),
            split=split, positive_label=positive_label, ax=ax,
        )

    def explanations(
        self,
        *,
        methods: Optional[Sequence[Any]] = None,
        model_name: Optional[str] = None,
        output_dir: str | Path = "generated_explanation",
        target: int = 1,
        method_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        show_checks: bool = True,
    ) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
        """Generate method-level XAI tables and combine them into one table."""
        explanation_iv_config = self._iv_config_for_explanations(methods)
        resolved_methods = self._xai_methods_from_iv_config(explanation_iv_config)
        if not resolved_methods:
            raise RuntimeError(
                "No XAI methods were provided and no `xai_method`/`xai_type` IV is stored. "
                "Call `add_iv('xai_method', ..., [...])` or pass `methods=[...]`."
            )

        if methods is None and show_checks:
            print(f"Using stored XAI methods from the design: {resolved_methods}")

        if model_name is None:
            model_name = self.model_name or "model"
            if show_checks:
                print(f"Using stored model name for explanation files: {model_name!r}")

        self.validate(stage="explanation_generation", show=show_checks)
        data = self._require_data()
        trained_ai_model = self._require_trained_ai_model()
        explanation_instance_ids = self._trial_ids_requiring_explanations()
        explanation_ids_by_method = (
            self._trial_ids_requiring_explanations_by_method()
        )
        if self.ai_predictions_by_instance is None:
            predictions = xai_adapter_api.predict_labels(
                trained_ai_model,
                data.split.X_model,
            )
            self.ai_predictions_by_instance = {
                int(instance_id): _as_python_scalar(prediction)
                for instance_id, prediction in zip(
                    data.split.raw_instance_ids,
                    predictions,
                )
            }

        self.explanation_config = xai_adapter_api.init_explanation_run(
            data=data,
            iv_config=explanation_iv_config,
            trained_ai_model=trained_ai_model,
            model_name=model_name,
            output_dir=self._resolve_output_path(output_dir),
            target=target,
            method_kwargs=method_kwargs,
            instance_ids=explanation_instance_ids,
            instance_ids_by_method=explanation_ids_by_method,
            predictions_by_instance=self.ai_predictions_by_instance,
        )
        self.explanation_paths, self.explanation_dfs = xai_adapter_api.generate_xai_explanation_tables(
            self.explanation_config
        )
        self.prediction_table_path, self.prediction_table = (
            xai_adapter_api.generate_ai_prediction_table(
                self.explanation_config
            )
        )
        self.explanation_paths.insert(0, self.prediction_table_path)
        self.explanation_dfs.insert(0, self.prediction_table)
        self.combined_explanation_path, self.combined_explanations = xai_adapter_api.combine_explanation_tables(
            self.explanation_dfs,
            self.explanation_config,
        )
        return self.combined_explanation_path, self.combined_explanations

    def plot_explanation(
        self,
        *,
        visualization: str = "influence",
        method: Optional[str] = None,
        instance_id: Optional[int] = None,
        top_n: int = 5,
        class_labels: Optional[Sequence[str]] = None,
        phase: Optional[str] = None,
        show_ai_prediction: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize one local explanation using the XAI adapter plot helper."""
        data = self._require_data()
        combined_df = self._require_combined_explanations()
        from src.xai_adapter import plot_explanation_visual

        if instance_id is None:
            candidate_rows = combined_df
            if "expMethod" in candidate_rows:
                if method is None:
                    candidate_rows = candidate_rows[
                        ~candidate_rows["expMethod"].astype(str).str.lower().isin(
                            {"__prediction_only__", "none", "no_xai", "control"}
                        )
                    ]
                else:
                    candidate_rows = candidate_rows[
                        candidate_rows["expMethod"].astype(str).str.lower().eq(
                            str(method).lower()
                        )
                    ]
            explanation_columns = [
                column
                for column in candidate_rows
                if column.startswith("a")
                and column.endswith("_i")
                and column[1:-2].isdigit()
            ]
            if explanation_columns:
                candidate_rows = candidate_rows[
                    candidate_rows[explanation_columns].notna().any(axis=1)
                ]
            if candidate_rows.empty:
                requested = f" for method {method!r}" if method is not None else ""
                raise ValueError(
                    "No generated explanation is available"
                    f"{requested}. Generate explanations for an XAI-visible "
                    "training or testing trial first."
                )
            instance_id = int(candidate_rows.iloc[0]["instanceId"])

        if show_ai_prediction is None:
            show_ai_prediction = (
                str(phase).lower() != "testing" if phase is not None else True
            )

        return plot_explanation_visual(
            combined_df,
            data,
            visualization=visualization,
            method=method,
            instance_id=instance_id,
            feature_names=data.raw_feature_names,
            top_n=top_n,
            class_labels=class_labels,
            show_ai_prediction=show_ai_prediction,
            **kwargs,
        )

    def preview_participant_trials(
        self,
        *,
        participant_id: int = 1,
        visualization: str = "importance",
        top_n: int = 5,
        class_labels: Optional[Sequence[str]] = None,
        fallback: str = "auto",
    ) -> Any:
        """Interactively preview one participant's trials with Back/Next controls."""
        data = self._require_data()
        trials = self._require_trials()
        pool = ensure_prediction_coverage(
            self._require_combined_explanations(),
            trials=trials,
            data=data,
            trained_ai_model=self._require_trained_ai_model(),
            model_name=self.model_name or "model",
            show=False,
        )
        return ep_preview.preview_participant_trials(
            data, trials, pool,
            participant_id=participant_id, visualization=visualization,
            top_n=top_n, class_labels=class_labels, fallback=fallback,
        )

    def preview_experiment_walkthrough(
        self,
        *,
        participant_id: int = 1,
        explanation_pool: Optional[pd.DataFrame] = None,
        visualization: str = "importance",
        top_n: int = 5,
        class_labels: Optional[Sequence[str]] = None,
        max_trials: Optional[int] = None,
        fallback: str = "auto",
    ) -> list[dict[str, Any]]:
        """Preview the full participant journey and expose final approval controls."""
        data = self._require_data()
        trials = self._require_trials()
        pool = explanation_pool if explanation_pool is not None else self._require_combined_explanations()
        pool = ensure_prediction_coverage(
            pool,
            trials=trials,
            data=data,
            trained_ai_model=self._require_trained_ai_model(),
            model_name=self.model_name or "model",
            show=False,
        )
        self.walkthrough_previewed = True
        self.walkthrough_approved = False
        return ep_preview.preview_experiment_walkthrough(
            self.study_protocol, data, trials, pool,
            participant_id=participant_id, visualization=visualization,
            top_n=top_n, class_labels=class_labels, max_trials=max_trials,
            on_approve=lambda: setattr(self, "walkthrough_approved", True),
            fallback=fallback,
        )

    def set_cognitive_model(
        self,
        cognitive_model: Optional[Callable[..., dict[str, Any]]] = None,
        *,
        cognitive_model_id: Optional[str] = None,
        cognitive_params: Optional[dict[str, float]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> "xaikitTest":
        """Store the cognitive model callable and parameter dictionary."""
        model_id = str(cognitive_model_id or "").lower().strip().replace("-", "_")
        if cognitive_model is not None and model_kwargs:
            raise ValueError("model_kwargs can only be used with a cognitive_model_id.")
        is_baseline = cognitive_models_api.is_baseline_model_id(model_id)
        if is_baseline:
            model_id = cognitive_models_api.normalize_baseline_model_id(model_id)
        if cognitive_model is None and is_baseline:
            cognitive_model = cognitive_models_api.create_baseline_model(
                model_id,
                **(model_kwargs or {}),
            )

        self.cognitive_model = cognitive_model or dummy_cognitive_model
        if cognitive_model_id is not None:
            self.cognitive_model_id = model_id
        elif cognitive_model is not None and self.cognitive_model_id == "placeholder":
            self.cognitive_model_id = "custom"
        self.cognitive_params = (
            ({} if is_baseline else default_cognitive_params())
            if cognitive_params is None
            else deepcopy(cognitive_params)
        )
        return self

    def run_experiment(
        self,
        *,
        mode: str = "participant_by_participant",
        participant_id: Optional[int] = 1,
        condition_filter: Optional[dict[str, Any]] = None,
        explanation_pool: Optional[pd.DataFrame] = None,
        require_walkthrough_approval: bool = False,
    ) -> pd.DataFrame:
        """Run the cognitive simulation over selected generated trial rows."""
        if require_walkthrough_approval and not self.walkthrough_approved:
            raise RuntimeError(
                "Experiment execution is locked. Preview the complete walkthrough and "
                "click 'Approve walkthrough' first."
            )
        data = self._require_data()
        trials = self._require_trials()
        explanation_pool = (
            explanation_pool
            if explanation_pool is not None
            else self._require_combined_explanations()
        )
        explanation_pool = ensure_prediction_coverage(
            explanation_pool,
            trials=trials,
            data=data,
            trained_ai_model=self._require_trained_ai_model(),
            model_name=self.model_name or "model",
        )

        self.simulated_results = virtual_experiment_api.run_experiment_executor(
            trials=trials,
            cognitive_params=self.cognitive_params,
            dvs=self.DVs,
            raw_dataset=data.df,
            explanation_pool=explanation_pool,
            mode=mode,
            participant_id=participant_id,
            condition_filter=condition_filter,
            condition_columns=[
                name
                for name, config in self.iv_config.items()
                if config.get("randomization") != "trial"
            ],
            cognitive_model=self.cognitive_model,
            label_column=data.label_column,
        )
        return self.simulated_results

    def save_results(
        self,
        *,
        out_dir: str | Path = "experiment_output",
    ) -> tuple[str, str]:
        """Save simulated experiment results as CSV and JSON."""
        if self.simulated_results is None:
            raise RuntimeError("Call run_experiment(...) before save_results(...).")
        self.simulated_csv_path, self.simulated_json_path = virtual_experiment_api.save_simulated_results(
            self.simulated_results,
            out_dir=self._resolve_output_path(out_dir),
        )
        return self.simulated_csv_path, self.simulated_json_path

    def analyze_iv_dv(
        self,
        *,
        iv: str,
        dv: str,
        participant_column: str = "participantId",
    ) -> Any:
        """Analyze one stored DV against one IV using testing responses."""
        if self.simulated_results is None:
            raise RuntimeError("Call run_experiment(...) before analyze_iv_dv(...).")
        from src.statistical_analyst import analyze_iv_dv

        return analyze_iv_dv(
            self.simulated_results,
            iv=iv,
            dv=dv,
            participant_column=participant_column,
        )

    def plot_results_grid(
        self,
        *,
        responses: Optional[pd.DataFrame] = None,
        ivs: Optional[Sequence[str]] = None,
        dvs: Optional[Sequence[str]] = None,
        participant_column: str = "participantId",
        phase: Optional[str] = "testing",
        errorbar: Optional[str] = "sem",
        title: Optional[str] = "Experiment results",
        value_labels: bool = True,
    ) -> Any:
        """Plot every requested dependent variable against every requested IV."""
        result_data = responses if responses is not None else self.simulated_results
        if result_data is None:
            raise RuntimeError("Call run_experiment(...) before plot_results_grid(...).")
        from src.result_visualizer import plot_iv_dv_grid

        resolved_ivs = list(ivs) if ivs is not None else list(self.iv_config)
        resolved_dvs = list(dvs) if dvs is not None else list(self.DVs)
        return plot_iv_dv_grid(
            result_data,
            ivs=resolved_ivs,
            dvs=resolved_dvs,
            participant_column=participant_column,
            phase=phase,
            errorbar=errorbar,
            iv_levels={
                name: config.get("levels", [])
                for name, config in self.iv_config.items()
                if name in resolved_ivs
            },
            title=title,
            value_labels=value_labels,
        )

    def plot_dv_by_two_ivs(
        self,
        *,
        x_iv: str,
        hue_iv: str,
        dv: str,
        responses: Optional[pd.DataFrame] = None,
        participant_column: str = "participantId",
        phase: Optional[str] = "testing",
        errorbar: Optional[str] = "sem",
        x_levels: Optional[Sequence[Any]] = None,
        hue_levels: Optional[Sequence[Any]] = None,
        x_labels: Optional[dict[Any, str]] = None,
        hue_labels: Optional[dict[Any, str]] = None,
        title: Optional[str] = None,
        value_labels: bool = True,
    ) -> Any:
        """Plot one DV against two IVs as grouped participant-level means."""
        result_data = responses if responses is not None else self.simulated_results
        if result_data is None:
            raise RuntimeError(
                "Call run_experiment(...) before plot_dv_by_two_ivs(...)."
            )
        from src.result_visualizer import (
            plot_dv_by_two_ivs as plot_interaction,
        )

        return plot_interaction(
            result_data,
            x_iv=x_iv,
            hue_iv=hue_iv,
            dv=dv,
            participant_column=participant_column,
            phase=phase,
            errorbar=errorbar,
            x_levels=(
                x_levels
                if x_levels is not None
                else self.iv_config.get(x_iv, {}).get("levels")
            ),
            hue_levels=(
                hue_levels
                if hue_levels is not None
                else self.iv_config.get(hue_iv, {}).get("levels")
            ),
            x_labels=x_labels,
            hue_labels=hue_labels,
            title=title,
            value_labels=value_labels,
        )

    def _resolve_output_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.output_dir / path

    def _iv_config_for_explanations(
        self,
        methods: Optional[Sequence[Any]],
    ) -> dict[str, dict[str, Any]]:
        iv_config = deepcopy(self.iv_config)
        if methods is None:
            return iv_config

        if "xai_method" in iv_config:
            iv_config["xai_method"]["levels"] = list(methods)
        elif "xai_type" in iv_config:
            iv_config["xai_type"]["levels"] = list(methods)
        else:
            iv_config["xai_method"] = {
                "type": "between",
                "levels": list(methods),
            }
        return iv_config

    def _xai_methods_from_iv_config(self, iv_config: dict[str, dict[str, Any]]) -> list[Any]:
        """Return stored XAI method/type levels from an IV config."""
        if "xai_method" in iv_config:
            return list(iv_config["xai_method"].get("levels", []))
        if "xai_type" in iv_config:
            return list(iv_config["xai_type"].get("levels", []))
        return []

    def _trial_ids_requiring_explanations(self) -> Optional[list[int]]:
        """Return training and XAI-visible testing IDs from generated trials."""
        if not self.trials:
            return None

        trials = pd.DataFrame(self.trials)
        if "instanceId" not in trials:
            return None

        method_column = (
            "xai_method"
            if "xai_method" in trials
            else "xai_type" if "xai_type" in trials else None
        )
        if method_column is None:
            method_has_xai = pd.Series(True, index=trials.index)
        else:
            method_has_xai = ~trials[method_column].astype(str).str.lower().isin(
                {"none", "no_xai", "control"}
            )

        training = (
            trials["phase"].astype(str).str.lower().eq("training")
            if "phase" in trials
            else pd.Series(False, index=trials.index)
        )
        if "tested_w_xai" in trials:
            tested_with_xai = trials["tested_w_xai"].map(
                lambda value: (
                    value.strip().lower() in {"true", "1", "yes", "y"}
                    if isinstance(value, str)
                    else bool(value)
                )
            )
        else:
            tested_with_xai = pd.Series(True, index=trials.index)

        required = trials[method_has_xai & (training | tested_with_xai)]
        return list(dict.fromkeys(required["instanceId"].astype(int).tolist()))

    def _trial_ids_requiring_explanations_by_method(
        self,
    ) -> Optional[dict[str, list[int]]]:
        """Return sampled XAI-visible instance IDs separately for each method."""
        if not self.trials:
            return None

        trials = pd.DataFrame(self.trials)
        if "instanceId" not in trials:
            return None
        method_column = (
            "xai_method"
            if "xai_method" in trials
            else "xai_type" if "xai_type" in trials else None
        )
        if method_column is None:
            return None

        methods = trials[method_column].astype(str).str.lower()
        method_has_xai = ~methods.isin({"none", "no_xai", "control"})
        training = (
            trials["phase"].astype(str).str.lower().eq("training")
            if "phase" in trials
            else pd.Series(False, index=trials.index)
        )
        if "tested_w_xai" in trials:
            tested_with_xai = trials["tested_w_xai"].map(
                lambda value: (
                    value.strip().lower() in {"true", "1", "yes", "y"}
                    if isinstance(value, str)
                    else bool(value)
                )
            )
        else:
            tested_with_xai = pd.Series(True, index=trials.index)

        required = trials[method_has_xai & (training | tested_with_xai)].copy()
        required["_method_key"] = methods.loc[required.index]
        return {
            method: list(dict.fromkeys(group["instanceId"].astype(int).tolist()))
            for method, group in required.groupby("_method_key", sort=False)
        }

    def _require_data(self) -> PreparedDataset:
        if self.data is None:
            raise RuntimeError("Call prepare_dataset(...) before this step.")
        return self.data

    def _require_model_manager(self) -> Any:
        if self.model_manager is None:
            raise RuntimeError("Call train_AI_model(...) before this step.")
        return self.model_manager

    def _require_trained_ai_model(self) -> Any:
        if self.trained_ai_model is None:
            raise RuntimeError("Call train_AI_model(...) before generating explanations.")
        return self.trained_ai_model

    def _require_trials(self) -> list[dict[str, Any]]:
        if not self.trials:
            raise RuntimeError("Call generate_trials(...) before running the experiment.")
        return self.trials

    def _require_combined_explanations(self) -> pd.DataFrame:
        if self.combined_explanations is None:
            raise RuntimeError("Call explanations(...) before this step.")
        return self.combined_explanations


_GUIDE_MESSAGES = {
    "design": (
        "Design guide\n"
        "Goal: decide what XAI methods you want to test and how the study compares them.\n"
        "IV: what you manipulate, e.g. `xai_method = ['shap', 'lime', 'none']`.\n"
        "CV: trial/participant metadata you control or record, e.g. age group, gender, user_task.\n"
        "DV: what you measure, e.g. `forward_accuracy`.\n"
        "User task: what participants/cognitive agents do, e.g. `forward_simulation` means predict the AI output from the instance and explanation.\n"
        "Typical call: add IVs, add CVs, add `user_task`, add DVs, then `validate(stage='design')`."
    ),
    "dataset": (
        "Dataset guide\n"
        "Goal: choose the dataset and feature subset used for model training, trials, and displays.\n"
        "Key args: `dataset_id`, optional `feature_cols` or defaults, `test_size`, `random_state`.\n"
        "XAIKit keeps raw values for display and model-ready values for training."
    ),
    "trial_generation": (
        "Trial guide\n"
        "Goal: sample training rows first and held-out testing rows second.\n"
        "For a machine-proxy study, train the AI first and pass "
        "`balance_by_ai_prediction=True` to sample each phase equally from its "
        "two predicted classes.\n"
        "The predicted-class order is randomized within each phase; the phase "
        "order is never counterbalanced."
    ),
    "model_training": (
        "Model guide\n"
        "Goal: train the AI model that later provides predictions and explanations.\n"
        "Supported model types: `mlp`, `xgboost`, `sim2real`.\n"
        "Key args: `model_type`, target metric/score, epoch limits, batch size."
    ),
    "explanation_generation": (
        "Explanation guide\n"
        "Goal: generate XAI tables for the methods stored in your design.\n"
        "Default methods come from `xai_method`; default model name comes from training.\n"
        "Key args: `target`, `output_dir`, optional method kwargs such as SHAP background size or LIME samples."
    ),
    "cognitive_models": (
        "Cognitive model guide\n"
        "Use the returned table to choose an agent and parameter ranges.\n"
        "Machine-proxy ids: `knn`, `decision_tree`, `logistic_regression`, `mlp`.\n"
        "Configure those with `model_kwargs`; configure cognitive agents with "
        "`cognitive_params`."
    ),
    "cognitive_simulation": (
        "Cognitive simulation guide\n"
        "Goal: run a cognitive model over generated trials.\n"
        "Use this when you want simulated behavior; planner-only workflows can skip it.\n"
        "Requires trials, AI predictions, and supported `user_task`/DV choices."
    ),
}


def _cognitive_model_guide(test: xaikitTest) -> str:
    """Return the cognitive-model guide plus current design compatibility."""
    lines = [_GUIDE_MESSAGES["cognitive_models"]]
    xai_kind, xai_values = _current_xai_design_values(test.iv_config)
    if not xai_values:
        lines.append("Current design: no `xai_method` or `xai_type` set yet.")
        return "\n".join(lines)

    support = load_support_matrix()
    compatible = []
    for agent in (
        "knn",
        "decision_tree",
        "logistic_regression",
        "mlp_baseline",
        "coax",
        "coxam",
        "sim2real",
    ):
        spec = support["cognitive_models"][agent]
        allowed = set(spec["xai_methods" if xai_kind == "xai_method" else "xai_types"])
        requested = {str(value).lower() for value in xai_values if str(value).lower() != "none"}
        if requested <= allowed:
            compatible.append(agent)

    lines.append(f"Current design: `{xai_kind}` = {_format_inline_values(xai_values)}.")
    lines.append(
        "Compatible cognitive agents: "
        f"{_format_inline_values(compatible) if compatible else 'none for the current XAI choices'}."
    )
    if xai_kind == "xai_method":
        lines.append("Note: `coax` supports attribution methods such as SHAP, LIME, LRP, IG, and DeepLift.")
    return "\n".join(lines)


def cognitive_model_guide_table() -> pd.DataFrame:
    """Return a notebook-friendly cognitive-model guide table."""
    return pd.DataFrame(
        {
            "KNN baseline": [
                "Nearest-example machine proxy",
                "", "", "", "", "", "", "", "",
            ],
            "Decision tree baseline": [
                "Rule-partition machine proxy",
                "", "", "", "", "", "", "", "",
            ],
            "Logistic baseline": [
                "Linear-boundary machine proxy",
                "", "", "", "", "", "", "", "",
            ],
            "MLP baseline": [
                "Nonlinear neural machine proxy",
                "", "", "", "", "", "", "", "",
            ],
            "CoAX": [
                "Attribution-method forward simulation",
                "[-2.3, -1.5] memory strictness",
                "[1, 10] similarity sensitivity",
                "[1, 5] attention span",
                "[1, 7] attribution-to-class strength",
                "",
                "",
                "",
                "",
            ],
            "CoXAM": [
                "Surrogate/strategy simulation for forward or counterfactual tasks",
                "[-2.8, -1.5] memory access",
                "",
                "",
                "",
                "[0, 10] accuracy-time tradeoff",
                "[0, 1] stochasticity",
                "[0, 1] counterfactual threshold",
                "",
            ],
            "Sim2Real": [
                "Feature-budget transfer tests",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "top_2_features or all_features",
            ],
        },
        index=[
            "Best for",
            "retrieval_threshold",
            "exemplar_distance_sensitivity",
            "attended_features",
            "feature_class_sensitivity",
            "opportunity_cost",
            "diffusion_noise",
            "counterfactual_margin",
            "memory_budget",
        ],
    )


def _current_xai_design_values(iv_config: dict[str, dict[str, Any]]) -> tuple[str, list[Any]]:
    if "xai_method" in iv_config:
        return "xai_method", list(iv_config["xai_method"].get("levels", []))
    if "xai_type" in iv_config:
        return "xai_type", list(iv_config["xai_type"].get("levels", []))
    return "xai_method", []


def _format_inline_values(values: Sequence[Any]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _as_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar predictions into JSON-safe Python values."""
    return value.item() if isinstance(value, np.generic) else value


XAIKitTest = xaikitTest


__all__ = [
    "XAIKitTest",
    "xaikitTest",
    "cognitive_model_guide_table",
]

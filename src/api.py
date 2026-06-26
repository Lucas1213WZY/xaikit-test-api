"""High-level XAIKit workflow API."""

from __future__ import annotations

import io
import base64
import html
import uuid
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.cognitive_models import default_cognitive_params, dummy_cognitive_model
from src.data_loaders import PreparedDataset, prepare_dataset, reencode_prepared_dataset
from src.experiment_design import (
    TrialGenerationResult,
    ValidationReport,
    generate_experimental_trials,
    init_experiment_config,
    init_trial_build_config,
    select_trial_rows,
    load_support_matrix,
    set_factor,
    set_iv,
    validate_experiment_config,
    validate_xaikit_test,
)
from src.virtual_experiment_executor import run_experiment_executor, save_simulated_results
from src.workflow_standard import (
    DEFAULT_EXPLANATION_INSTANCE_LIMIT,
    EXPLANATION_METHOD_COL,
    INSTANCE_ID_COL,
    PREDICTION_COL,
    ensure_prediction_coverage,
)
import src.xai_adapter as xai_adapter_api


class XAIKitTest:
    """Collate one XAI experiment workflow into a single reusable object."""

    def __init__(
        self,
        project_name: str = "xaikit_test",
        *,
        output_dir: str | Path = ".",
        auto_validate_design: bool = True,
        show_greeting: bool = True,
    ) -> None:
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.auto_validate_design = auto_validate_design
        self.show_greeting = show_greeting

        self.iv_config, self.CVs, self.DVs = init_experiment_config()

        self.data: Optional[PreparedDataset] = None
        self.model_manager = None
        self.model = None
        self.trained_engine = None
        self.model_name: Optional[str] = None
        self.model_source: Optional[str] = None
        self.training_info: Optional[dict[str, Any]] = None
        self.training_stdout: str = ""
        self.metrics: dict[str, dict[str, Any]] = {}

        self.trial_config = None
        self.trial_result: Optional[TrialGenerationResult] = None
        self.trials: list[dict[str, Any]] = []

        self.explanation_config = None
        self.explanation_paths: list[Path] = []
        self.explanation_dfs: list[pd.DataFrame] = []
        self.combined_explanation_path: Optional[Path] = None
        self.combined_explanations: Optional[pd.DataFrame] = None

        self.cognitive_params: dict[str, float] = default_cognitive_params()
        self.cognitive_model: Callable[..., dict[str, Any]] = dummy_cognitive_model
        self.cognitive_model_id: str = "placeholder"
        self.validation_reports: dict[str, ValidationReport] = {}
        self.simulated_results: Optional[pd.DataFrame] = None
        self.simulated_csv_path: Optional[str] = None
        self.simulated_json_path: Optional[str] = None

        if self.show_greeting:
            print(_workflow_greeting(self.project_name))

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
    ) -> "XAIKitTest":
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
    ) -> "XAIKitTest":
        """Add or replace one independent variable."""
        set_iv(self.iv_config, name, iv_type, levels, randomization=randomization)
        if show:
            self.validate_design(show=True)
        return self

    def add_cv(self, name: str, levels: list[Any], *, show: bool = False) -> "XAIKitTest":
        """Add or replace one control variable."""
        set_factor(self.CVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def add_dv(self, name: str, levels: list[Any], *, show: bool = False) -> "XAIKitTest":
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
        participants_per_between_condition: int = 25,
        trials_per_participant: int = 10,
        trial_randomization_strategy: str = "balanced",
        instance_wise_explanation: bool = False,
        shuffle_instances: bool = True,
        max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT,
        seed: int = 42,
        output_dir: str | Path = "experiment_output",
        preview_rows: int = 10,
        show: bool = True,
    ) -> TrialGenerationResult:
        """Build, export, and preview experiment trial rows."""
        data = self._require_data()
        if model_name is not None:
            self.model_name = model_name
        self.trial_config = init_trial_build_config(
            data=data,
            iv_config=self.iv_config,
            cvs=self.CVs,
            model_name=model_name,
            participants_per_between_condition=participants_per_between_condition,
            trials_per_participant=trials_per_participant,
            trial_randomization_strategy=trial_randomization_strategy,
            instance_wise_explanation=instance_wise_explanation,
            shuffle_instances=shuffle_instances,
            max_trial_instances=max_trial_instances,
            seed=seed,
            output_dir=self._resolve_output_path(output_dir),
        )
        self.trial_result = generate_experimental_trials(
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

        self.trained_engine = getattr(self.model, "engine", self.model)
        self.model_source = source
        return self.model

    def train_ai_model(self, **kwargs: Any) -> Any:
        """PEP-8 alias for `train_AI_model`."""
        return self.train_AI_model(**kwargs)

    def training_summary_table(self) -> pd.DataFrame:
        """Return a one-row summary of the latest training run."""
        if self.training_info is None:
            raise RuntimeError("Call train_AI_model(...) before requesting the training summary.")

        summary = {
            key: value
            for key, value in self.training_info.items()
            if key != "history"
        }
        summary["model_type"] = self.model_name
        summary["dataset"] = self.data.dataset_id if self.data is not None else None
        return pd.DataFrame([summary])

    def training_history_table(self) -> pd.DataFrame:
        """Return the accuracy checkpoint history from the latest training run."""
        if self.training_info is None:
            raise RuntimeError("Call train_AI_model(...) before requesting training history.")

        history = self.training_info.get("history", [])
        if not history:
            return self.training_summary_table()

        history_df = pd.DataFrame(history)
        target_metric = self.training_info.get("target_metric", "accuracy")
        target_score = self.training_info.get(
            "target_score",
            self.training_info.get("target_accuracy"),
        )
        if target_score is not None:
            history_df["target_metric"] = target_metric
            history_df["target_score"] = target_score
            history_df["reached_target"] = history_df[target_metric] >= target_score
            if target_metric == "accuracy":
                history_df["target_accuracy"] = target_score
        return history_df

    def plot_training_history(self, *, ax: Any = None) -> Any:
        """Plot metric checkpoints from training with a target score."""
        history_df = self.training_history_table()
        target_metric = (
            self.training_info.get("target_metric", "accuracy")
            if self.training_info is not None
            else "accuracy"
        )
        if "epochs" not in history_df or target_metric not in history_df:
            raise RuntimeError(f"Training history does not contain epoch-level {target_metric} checkpoints.")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.plot(history_df["epochs"], history_df[target_metric], marker="o", label=f"Training {target_metric}")
        if "target_score" in history_df:
            target_score = float(history_df["target_score"].iloc[0])
            ax.axhline(target_score, color="black", linestyle="--", linewidth=1, label="Target")
        ax.set_title(f"Training {target_metric} Checkpoints")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(target_metric)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

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
        data = self._require_data()
        manager = self._require_model_manager()
        split = split.lower()

        if split == "train":
            X, y = data.X_train, data.y_train
        elif split == "test":
            X, y = data.X_test, data.y_test
        else:
            raise ValueError("split must be one of: 'train' or 'test'.")

        metrics = manager.evaluate_metrics(
            X,
            y,
            positive_label=positive_label,
            threshold=threshold,
        )
        labels = metrics["labels"]
        return pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
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
        import matplotlib.pyplot as plt

        matrix_df = self.confusion_matrix_table(
            split=split,
            positive_label=positive_label,
            threshold=threshold,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(4.5, 4))

        image = ax.imshow(matrix_df.to_numpy(), cmap="Blues")
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix_df.index)), matrix_df.index)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title(f"{split.title()} Confusion Matrix")

        for row_index, row in enumerate(matrix_df.to_numpy()):
            for col_index, value in enumerate(row):
                ax.text(col_index, row_index, int(value), ha="center", va="center")
        return ax

    def evaluate(
        self,
        *,
        split: str = "both",
        positive_label: int = 1,
        threshold: float = 0.5,
        include_report: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate the trained AI model with classic classification metrics."""
        data = self._require_data()
        manager = self._require_model_manager()
        split = split.lower()
        results: dict[str, dict[str, Any]] = {}

        if split in {"train", "both"}:
            results["train"] = manager.evaluate_metrics(
                data.X_train,
                data.y_train,
                positive_label=positive_label,
                threshold=threshold,
                include_report=include_report,
            )
        if split in {"test", "both"}:
            results["test"] = manager.evaluate_metrics(
                data.X_test,
                data.y_test,
                positive_label=positive_label,
                threshold=threshold,
                include_report=include_report,
            )
        if not results:
            raise ValueError("split must be one of: 'train', 'test', or 'both'.")

        self.metrics.update(results)
        return results

    def metrics_table(self) -> pd.DataFrame:
        """Return scalar evaluation metrics as a compact split-by-metric table."""
        if not self.metrics:
            raise RuntimeError("Call evaluate(...) before requesting the metrics table.")

        preferred_columns = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
        rows = {}
        for split, metrics in self.metrics.items():
            rows[split] = {
                key: value
                for key, value in metrics.items()
                if np.isscalar(value) or value is None
            }

        metrics_df = pd.DataFrame.from_dict(rows, orient="index")
        ordered_columns = [
            column
            for column in preferred_columns
            if column in metrics_df.columns
        ]
        ordered_columns.extend(
            column
            for column in metrics_df.columns
            if column not in ordered_columns
        )
        return metrics_df.loc[:, ordered_columns]

    def plot_auc_curves(
        self,
        *,
        split: str = "both",
        positive_label: int = 1,
        ax: Any = None,
    ) -> Any:
        """Plot ROC curves and AUC values for train/test predictions."""
        data = self._require_data()
        manager = self._require_model_manager()
        from sklearn.metrics import auc, roc_curve
        import matplotlib.pyplot as plt

        split = split.lower()
        split_data = []
        if split in {"train", "both"}:
            split_data.append(("train", data.X_train, data.y_train))
        if split in {"test", "both"}:
            split_data.append(("test", data.X_test, data.y_test))
        if not split_data:
            raise ValueError("split must be one of: 'train', 'test', or 'both'.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        for split_name, X, y in split_data:
            scores = self._positive_class_scores(
                manager.predict(X),
                y,
                positive_label=positive_label,
            )
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _thresholds = roc_curve(y, scores, pos_label=positive_label)
            ax.plot(fpr, tpr, linewidth=2, label=f"{split_name} AUC={auc(fpr, tpr):.3f}")

        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="chance")
        ax.set_title("ROC-AUC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

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
        trained_engine = self._require_trained_engine()

        self.explanation_config = xai_adapter_api.init_explanation_run(
            data=data,
            iv_config=explanation_iv_config,
            trained_engine=trained_engine,
            model_name=model_name,
            output_dir=self._resolve_output_path(output_dir),
            target=target,
            method_kwargs=method_kwargs,
        )
        self.explanation_paths, self.explanation_dfs = xai_adapter_api.generate_xai_explanation_tables(
            self.explanation_config
        )
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
        **kwargs: Any,
    ) -> Any:
        """Visualize one local explanation using the XAI adapter plot helper."""
        data = self._require_data()
        combined_df = self._require_combined_explanations()
        from src.xai_adapter import plot_explanation_visual

        if instance_id is None:
            instance_id = int(data.test_instance_ids[0])

        return plot_explanation_visual(
            combined_df,
            data,
            visualization=visualization,
            method=method,
            instance_id=instance_id,
            feature_names=data.raw_feature_names,
            top_n=top_n,
            class_labels=class_labels,
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
        """
        Interactively preview one participant's trials with Back/Next controls.

        Trials with `tested_w_xai=False` or a no-XAI method show the raw instance
        and prediction context. Trials with XAI show the matching explanation plot.
        """
        data = self._require_data()
        trials = self._require_trials()
        combined_df = self._require_combined_explanations()
        participant_trials = select_trial_rows(
            pd.DataFrame(trials),
            mode="participant_by_participant",
            participant_id=participant_id,
        ).reset_index(drop=True)
        if participant_trials.empty:
            raise ValueError(f"No trials found for participant_id={participant_id}.")

        fallback = fallback.lower().strip()
        if fallback not in {"auto", "widgets", "html"}:
            raise ValueError("fallback must be one of: 'auto', 'widgets', or 'html'.")
        if fallback == "html":
            return self._display_participant_preview_html(
                participant_trials,
                combined_df,
                participant_id=participant_id,
                visualization=visualization,
                top_n=top_n,
                class_labels=class_labels,
            )

        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display
        except ImportError:
            if fallback == "widgets":
                print("Interactive preview requires `ipywidgets` and IPython display support.")
                return participant_trials
            return self._display_participant_preview_html(
                participant_trials,
                combined_df,
                participant_id=participant_id,
                visualization=visualization,
                top_n=top_n,
                class_labels=class_labels,
            )

        print("Use Back/Next to preview this participant's trials.")
        print("If widgets do not render in this notebook, call with `fallback='html'` after updating.")

        state = {"idx": 0}
        output = widgets.Output()
        back_button = widgets.Button(description="Back", icon="arrow-left")
        next_button = widgets.Button(description="Next", icon="arrow-right")
        label = widgets.HTML()

        def render() -> None:
            with output:
                clear_output(wait=True)
                trial = participant_trials.iloc[state["idx"]].to_dict()
                label.value = (
                    f"<b>Participant {participant_id}</b> | "
                    f"Trial {state['idx'] + 1} of {len(participant_trials)}"
                )
                self._display_trial_preview(
                    trial,
                    combined_df,
                    visualization=visualization,
                    top_n=top_n,
                    class_labels=class_labels,
                )
            back_button.disabled = state["idx"] == 0
            next_button.disabled = state["idx"] == len(participant_trials) - 1

        def go_back(_button: Any) -> None:
            state["idx"] = max(0, state["idx"] - 1)
            render()

        def go_next(_button: Any) -> None:
            state["idx"] = min(len(participant_trials) - 1, state["idx"] + 1)
            render()

        back_button.on_click(go_back)
        next_button.on_click(go_next)
        controls = widgets.HBox([back_button, next_button, label])
        render()
        display(widgets.VBox([controls, output]))
        return participant_trials

    def _display_participant_preview_html(
        self,
        participant_trials: pd.DataFrame,
        explanation_df: pd.DataFrame,
        *,
        participant_id: int,
        visualization: str,
        top_n: int,
        class_labels: Optional[Sequence[str]],
    ) -> Any:
        """Display an HTML Back/Next preview when ipywidgets is unavailable."""
        try:
            from IPython.display import HTML, display
        except ImportError:
            print("Interactive preview requires IPython display support.")
            return participant_trials

        preview_id = f"xaikit-preview-{uuid.uuid4().hex}"
        slides = [
            self._trial_preview_html(
                trial.to_dict(),
                explanation_df,
                slide_number=idx + 1,
                slide_count=len(participant_trials),
                visualization=visualization,
                top_n=top_n,
                class_labels=class_labels,
            )
            for idx, trial in participant_trials.iterrows()
        ]
        slides_html = "\n".join(
            f'<div class="xaikit-slide" data-index="{idx}" style="display: {"block" if idx == 0 else "none"};">'
            f"{slide}</div>"
            for idx, slide in enumerate(slides)
        )
        html_doc = f"""
        <div id="{preview_id}" class="xaikit-participant-preview">
          <style>
            #{preview_id} {{
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              color: #111827;
            }}
            #{preview_id} .xaikit-controls {{
              display: flex;
              align-items: center;
              gap: 8px;
              margin: 8px 0 12px;
            }}
            #{preview_id} button {{
              border: 1px solid #9ca3af;
              background: #ffffff;
              color: #111827;
              padding: 5px 10px;
              border-radius: 4px;
              cursor: pointer;
            }}
            #{preview_id} button:disabled {{
              color: #9ca3af;
              cursor: default;
            }}
            #{preview_id} .xaikit-count {{
              font-weight: 600;
              margin-left: 4px;
            }}
            #{preview_id} table {{
              border-collapse: collapse;
              margin: 6px 0 12px;
              font-size: 13px;
            }}
            #{preview_id} th, #{preview_id} td {{
              border: 1px solid #d1d5db;
              padding: 5px 8px;
              text-align: left;
            }}
            #{preview_id} th {{
              background: #f3f4f6;
            }}
            #{preview_id} img {{
              max-width: 100%;
              height: auto;
            }}
            #{preview_id} .xaikit-note {{
              margin: 8px 0;
              color: #374151;
            }}
          </style>
          <div class="xaikit-controls">
            <button type="button" class="xaikit-back">Back</button>
            <button type="button" class="xaikit-next">Next</button>
            <span class="xaikit-count"></span>
          </div>
          {slides_html}
          <script>
            (function() {{
              const root = document.getElementById("{preview_id}");
              const slides = Array.from(root.querySelectorAll(".xaikit-slide"));
              const back = root.querySelector(".xaikit-back");
              const next = root.querySelector(".xaikit-next");
              const count = root.querySelector(".xaikit-count");
              let index = 0;
              function render() {{
                slides.forEach((slide, i) => {{
                  slide.style.display = i === index ? "block" : "none";
                }});
                back.disabled = index === 0;
                next.disabled = index === slides.length - 1;
                count.textContent = "Participant {participant_id} | Trial " + (index + 1) + " of " + slides.length;
              }}
              back.addEventListener("click", function() {{
                index = Math.max(0, index - 1);
                render();
              }});
              next.addEventListener("click", function() {{
                index = Math.min(slides.length - 1, index + 1);
                render();
              }});
              render();
            }})();
          </script>
        </div>
        """
        print("ipywidgets unavailable. Showing HTML Back/Next preview.")
        display(HTML(html_doc))
        return participant_trials

    def _trial_preview_html(
        self,
        trial: dict[str, Any],
        explanation_df: pd.DataFrame,
        *,
        slide_number: int,
        slide_count: int,
        visualization: str,
        top_n: int,
        class_labels: Optional[Sequence[str]],
    ) -> str:
        """Render one trial preview as HTML for widget-free notebooks."""
        data = self._require_data()
        instance_id = int(trial["instanceId"])
        xai_method = str(trial.get("xai_method", trial.get("xai_type", "none"))).lower()
        tested_w_xai = _coerce_bool(
            trial.get(
                "tested_w_xai",
                xai_method not in {"none", "no_xai", "control"},
            )
        )
        has_xai = tested_w_xai and xai_method not in {"none", "no_xai", "control"}

        summary = pd.DataFrame([{
            "participantId": trial.get("participantId"),
            "trialId": trial.get("trialId"),
            "instanceId": instance_id,
            "xai_method": xai_method,
            "tested_w_xai": tested_w_xai,
            "ai_prediction": _prediction_for_instance(explanation_df, instance_id),
        }]).to_html(index=False, border=0)

        body = ""
        if has_xai:
            try:
                from src.xai_adapter import plot_explanation_visual
                import matplotlib.pyplot as plt

                fig, _axes = plot_explanation_visual(
                    explanation_df,
                    data,
                    visualization=visualization,
                    method=xai_method,
                    instance_id=instance_id,
                    feature_names=data.raw_feature_names,
                    top_n=top_n,
                    class_labels=list(class_labels) if class_labels is not None else None,
                )
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                image = base64.b64encode(buffer.getvalue()).decode("ascii")
                body = f'<img alt="Trial {slide_number} explanation" src="data:image/png;base64,{image}">'
            except Exception as exc:
                body = f'<p class="xaikit-note">Could not render this explanation: {html.escape(str(exc))}</p>'

        if not body:
            row = data.df.iloc[instance_id].drop(labels=[data.label_column], errors="ignore")
            raw_values = pd.DataFrame(row, columns=["value"]).T.to_html(index=False, border=0)
            body = '<p class="xaikit-note">No XAI shown for this trial.</p>' + raw_values

        return (
            f'<div class="xaikit-note">Trial {slide_number} of {slide_count}</div>'
            f"{summary}"
            f"{body}"
        )

    def _display_trial_preview(
        self,
        trial: dict[str, Any],
        explanation_df: pd.DataFrame,
        *,
        visualization: str,
        top_n: int,
        class_labels: Optional[Sequence[str]],
    ) -> None:
        """Display one trial as either an explanation plot or no-XAI instance preview."""
        from IPython.display import display

        data = self._require_data()
        instance_id = int(trial["instanceId"])
        xai_method = str(trial.get("xai_method", trial.get("xai_type", "none"))).lower()
        tested_w_xai = _coerce_bool(
            trial.get(
                "tested_w_xai",
                xai_method not in {"none", "no_xai", "control"},
            )
        )
        has_xai = tested_w_xai and xai_method not in {"none", "no_xai", "control"}

        summary = {
            "participantId": trial.get("participantId"),
            "trialId": trial.get("trialId"),
            "instanceId": instance_id,
            "xai_method": xai_method,
            "tested_w_xai": tested_w_xai,
            "ai_prediction": _prediction_for_instance(explanation_df, instance_id),
        }
        display(pd.DataFrame([summary]))

        if has_xai:
            from src.xai_adapter import plot_explanation_visual

            try:
                fig, _axes = plot_explanation_visual(
                    explanation_df,
                    data,
                    visualization=visualization,
                    method=xai_method,
                    instance_id=instance_id,
                    feature_names=data.raw_feature_names,
                    top_n=top_n,
                    class_labels=list(class_labels) if class_labels is not None else None,
                )
                display(fig)
                return
            except ValueError as exc:
                print(f"No generated explanation found for this trial: {exc}")

        print("No XAI shown for this trial.")
        row = data.df.iloc[instance_id].drop(labels=[data.label_column], errors="ignore")
        display(pd.DataFrame(row, columns=["value"]).T)

    def set_cognitive_model(
        self,
        cognitive_model: Optional[Callable[..., dict[str, Any]]] = None,
        *,
        cognitive_model_id: Optional[str] = None,
        cognitive_params: Optional[dict[str, float]] = None,
    ) -> "XAIKitTest":
        """Store the cognitive model callable and parameter dictionary."""
        self.cognitive_model = cognitive_model or dummy_cognitive_model
        if cognitive_model_id is not None:
            self.cognitive_model_id = str(cognitive_model_id)
        elif cognitive_model is not None and self.cognitive_model_id == "placeholder":
            self.cognitive_model_id = "custom"
        self.cognitive_params = (
            default_cognitive_params()
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
    ) -> pd.DataFrame:
        """Run the cognitive simulation over selected generated trial rows."""
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
            trained_engine=self._require_trained_engine(),
            model_name=self.model_name or "model",
        )

        self.simulated_results = run_experiment_executor(
            trials=trials,
            cognitive_params=self.cognitive_params,
            dvs=self.DVs,
            raw_dataset=data.df,
            explanation_pool=explanation_pool,
            mode=mode,
            participant_id=participant_id,
            condition_filter=condition_filter,
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
        self.simulated_csv_path, self.simulated_json_path = save_simulated_results(
            self.simulated_results,
            out_dir=self._resolve_output_path(out_dir),
        )
        return self.simulated_csv_path, self.simulated_json_path

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

    def _require_data(self) -> PreparedDataset:
        if self.data is None:
            raise RuntimeError("Call prepare_dataset(...) before this step.")
        return self.data

    def _require_model_manager(self) -> Any:
        if self.model_manager is None:
            raise RuntimeError("Call train_AI_model(...) before this step.")
        return self.model_manager

    def _require_trained_engine(self) -> Any:
        if self.trained_engine is None:
            raise RuntimeError("Call train_AI_model(...) before generating explanations.")
        return self.trained_engine

    def _require_trials(self) -> list[dict[str, Any]]:
        if not self.trials:
            raise RuntimeError("Call generate_trials(...) before running the experiment.")
        return self.trials

    def _require_combined_explanations(self) -> pd.DataFrame:
        if self.combined_explanations is None:
            raise RuntimeError("Call explanations(...) before this step.")
        return self.combined_explanations

    def _positive_class_scores(
        self,
        predictions: Any,
        y: Sequence[Any],
        *,
        positive_label: int = 1,
    ) -> np.ndarray:
        preds = np.asarray(predictions)
        labels = np.unique(np.asarray(y))

        if preds.ndim == 2:
            positive_index = list(labels).index(positive_label) if positive_label in labels else -1
            return preds[:, positive_index]

        flat = preds.reshape(-1)
        if np.issubdtype(flat.dtype, np.floating):
            return flat
        return (flat == positive_label).astype(float)


xaikitTest = XAIKitTest


def _workflow_greeting(project_name: str) -> str:
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    return (
        f"{greeting}. XAIKitTest is ready for '{project_name}'.\n"
        "Start by choosing the XAI experiment you want to run. "
        "Call `xaikitTest.guide_design()` for a concise design guide."
    )


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
        "Goal: create participant trial rows from your design and test instances.\n"
        "Key args: participants per condition, trials per participant, randomization strategy, seed.\n"
        "By default, trials use the same 300 test-instance pool used for explanations."
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
        "Typical call: `set_cognitive_model(cognitive_model_id='coax', cognitive_params={...})`."
    ),
    "cognitive_simulation": (
        "Cognitive simulation guide\n"
        "Goal: run a cognitive model over generated trials.\n"
        "Use this when you want simulated behavior; planner-only workflows can skip it.\n"
        "Requires trials, AI predictions, and supported `user_task`/DV choices."
    ),
}


def _cognitive_model_guide(test: XAIKitTest) -> str:
    """Return the cognitive-model guide plus current design compatibility."""
    lines = [_GUIDE_MESSAGES["cognitive_models"]]
    xai_kind, xai_values = _current_xai_design_values(test.iv_config)
    if not xai_values:
        lines.append("Current design: no `xai_method` or `xai_type` set yet.")
        return "\n".join(lines)

    support = load_support_matrix()
    compatible = []
    for agent in ("coax", "coxam", "sim2real"):
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


def _coerce_bool(value: Any) -> bool:
    """Interpret notebook/CSV boolean-like values consistently."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    if text in {"", "0", "false", "n", "nan", "no", "none"}:
        return False
    if text in {"1", "true", "y", "yes"}:
        return True
    return bool(value)


def _prediction_for_instance(explanation_df: pd.DataFrame, instance_id: int) -> Any:
    """Return the first stored AI prediction for an instance, if available."""
    required_cols = {INSTANCE_ID_COL, PREDICTION_COL}
    if not required_cols <= set(explanation_df.columns):
        return None

    matches = explanation_df[explanation_df[INSTANCE_ID_COL].astype(int) == int(instance_id)]
    if matches.empty:
        return None
    if EXPLANATION_METHOD_COL in matches:
        explanation_matches = matches[
            matches[EXPLANATION_METHOD_COL].astype(str).str.lower() != "none"
        ]
        if not explanation_matches.empty:
            matches = explanation_matches
    prediction = matches[PREDICTION_COL].dropna()
    if prediction.empty:
        return None
    value = prediction.iloc[0]
    try:
        numeric_value = float(value)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value
    except (TypeError, ValueError):
        return value


__all__ = [
    "XAIKitTest",
    "cognitive_model_guide_table",
    "xaikitTest",
]

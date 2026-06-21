"""High-level workflow object for XAIKit tutorial experiments.

`XAIKitTest` is a notebook-facing orchestration layer. It stores the repeated
workflow state once, then delegates the real work to the existing utilities and
`src` interfaces.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from utils import (
        PreparedDataset,
        TrialGenerationResult,
        combine_explanation_tables,
        default_cognitive_params,
        dummy_cognitive_model,
        ensure_project_imports,
        generate_experimental_trials,
        generate_xai_explanation_tables,
        init_experiment_config,
        init_explanation_run,
        init_trial_build_config,
        prepare_dataset,
        run_experiment_executor,
        save_simulated_results,
        set_factor,
        set_iv,
        validate_experiment_config,
    )
except ModuleNotFoundError:
    from tutorials.utils import (
        PreparedDataset,
        TrialGenerationResult,
        combine_explanation_tables,
        default_cognitive_params,
        dummy_cognitive_model,
        ensure_project_imports,
        generate_experimental_trials,
        generate_xai_explanation_tables,
        init_experiment_config,
        init_explanation_run,
        init_trial_build_config,
        prepare_dataset,
        run_experiment_executor,
        save_simulated_results,
        set_factor,
        set_iv,
        validate_experiment_config,
    )


class XAIKitTest:
    """Collate one XAI experiment workflow into a single reusable object."""

    def __init__(
        self,
        project_name: str = "xaikit_test",
        *,
        output_dir: str | Path = ".",
        auto_validate_design: bool = True,
    ) -> None:
        ensure_project_imports()
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.auto_validate_design = auto_validate_design

        self.iv_config, self.CVs, self.DVs = init_experiment_config()

        self.data: Optional[PreparedDataset] = None
        self.model_manager = None
        self.model = None
        self.trained_engine = None
        self.model_name: Optional[str] = None
        self.training_info: Optional[Dict[str, Any]] = None
        self.training_stdout: str = ""
        self.metrics: Dict[str, Dict[str, Any]] = {}

        self.trial_config = None
        self.trial_result: Optional[TrialGenerationResult] = None
        self.trials: List[Dict[str, Any]] = []

        self.explanation_config = None
        self.explanation_paths: List[Path] = []
        self.explanation_dfs: List[pd.DataFrame] = []
        self.combined_explanation_path: Optional[Path] = None
        self.combined_explanations: Optional[pd.DataFrame] = None

        self.cognitive_params: Dict[str, float] = default_cognitive_params()
        self.cognitive_model: Callable[..., Dict[str, Any]] = dummy_cognitive_model
        self.simulated_results: Optional[pd.DataFrame] = None
        self.simulated_csv_path: Optional[str] = None
        self.simulated_json_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Experiment design
    # ------------------------------------------------------------------

    def set_design(
        self,
        *,
        iv_config: Optional[Dict[str, Dict[str, Any]]] = None,
        cvs: Optional[Dict[str, List[Any]]] = None,
        dvs: Optional[Dict[str, List[Any]]] = None,
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
        levels: List[Any],
        *,
        randomization: str = "block",
        show: bool = False,
    ) -> "XAIKitTest":
        """Add or replace one independent variable."""
        set_iv(self.iv_config, name, iv_type, levels, randomization=randomization)
        if show:
            self.validate_design(show=True)
        return self

    def add_cv(self, name: str, levels: List[Any], *, show: bool = False) -> "XAIKitTest":
        """Add or replace one control variable."""
        set_factor(self.CVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def add_dv(self, name: str, levels: List[Any], *, show: bool = False) -> "XAIKitTest":
        """Add or replace one dependent variable."""
        set_factor(self.DVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def validate_design(self, *, show: bool = True) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Validate and optionally print the current experimental design."""
        return validate_experiment_config(self.iv_config, self.CVs, self.DVs, show=show)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def prepare_dataset(
        self,
        dataset_id: str,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        num_features: Optional[int] = None,
        rank_features_by_target: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        show_available: bool = True,
        show_summary: bool = True,
    ) -> PreparedDataset:
        """Load, optionally feature-select, and split the dataset."""
        self.data = prepare_dataset(
            dataset_id,
            feature_cols=feature_cols,
            num_features=num_features,
            rank_features_by_target=rank_features_by_target,
            test_size=test_size,
            random_state=random_state,
            show_available=show_available,
            show_summary=show_summary,
        )
        return self.data

    # ------------------------------------------------------------------
    # Trials
    # ------------------------------------------------------------------

    def generate_trials(
        self,
        *,
        model_name: str = "mlp",
        participants_per_between_condition: int = 25,
        trials_per_participant: int = 10,
        trial_randomization_strategy: str = "balanced",
        instance_wise_explanation: bool = False,
        shuffle_instances: bool = True,
        seed: int = 42,
        output_dir: str | Path = "experiment_output",
        preview_rows: int = 10,
        show: bool = True,
    ) -> TrialGenerationResult:
        """Build, export, and preview experiment trial rows."""
        data = self._require_data()
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

    # ------------------------------------------------------------------
    # AI model
    # ------------------------------------------------------------------

    def train_AI_model(
        self,
        *,
        model_type: str = "mlp",
        source: Optional[str] = None,
        target_accuracy: Optional[float] = None,
        max_epochs: int = 300,
        check_every_epochs: int = 10,
        batch_size: int = 1000,
        verbose: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create and train the AI model used for predictions and explanations."""
        data = self._require_data()
        ensure_project_imports()
        from src.ai_models import ModelManager

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

            if target_accuracy is None:
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
                    target_accuracy=target_accuracy,
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
        if "target_accuracy" in self.training_info:
            history_df["target_accuracy"] = self.training_info["target_accuracy"]
        history_df["reached_target"] = history_df["accuracy"] >= history_df.get(
            "target_accuracy",
            np.nan,
        )
        return history_df

    def plot_training_history(self, *, ax: Any = None) -> Any:
        """Plot accuracy checkpoints from `train_until_accuracy`."""
        history_df = self.training_history_table()
        if "epochs" not in history_df or "accuracy" not in history_df:
            raise RuntimeError("Training history does not contain epoch-level accuracy checkpoints.")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.plot(history_df["epochs"], history_df["accuracy"], marker="o", label="Training accuracy")
        if "target_accuracy" in history_df:
            target_accuracy = float(history_df["target_accuracy"].iloc[0])
            ax.axhline(target_accuracy, color="black", linestyle="--", linewidth=1, label="Target")
        ax.set_title("Training Accuracy Checkpoints")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

    def evaluate(
        self,
        *,
        split: str = "both",
        positive_label: int = 1,
        threshold: float = 0.5,
        include_report: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate the trained AI model with classic classification metrics."""
        data = self._require_data()
        manager = self._require_model_manager()
        split = split.lower()
        results: Dict[str, Dict[str, Any]] = {}

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

    # ------------------------------------------------------------------
    # Explanations and visualization
    # ------------------------------------------------------------------

    def explanations(
        self,
        *,
        methods: Optional[Sequence[Any]] = None,
        model_name: Optional[str] = None,
        output_dir: str | Path = "generated_explanation",
        target: int = 1,
        method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
        """Generate method-level XAI tables and combine them into one table."""
        data = self._require_data()
        trained_engine = self._require_trained_engine()
        explanation_iv_config = self._iv_config_for_explanations(methods)
        model_name = model_name or self.model_name or "model"

        self.explanation_config = init_explanation_run(
            data=data,
            iv_config=explanation_iv_config,
            trained_engine=trained_engine,
            model_name=model_name,
            output_dir=self._resolve_output_path(output_dir),
            target=target,
            method_kwargs=method_kwargs,
        )
        self.explanation_paths, self.explanation_dfs = generate_xai_explanation_tables(
            self.explanation_config
        )
        self.combined_explanation_path, self.combined_explanations = combine_explanation_tables(
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
            feature_names=data.feature_names,
            top_n=top_n,
            class_labels=class_labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Cognitive model and experiment simulation
    # ------------------------------------------------------------------

    def set_cognitive_model(
        self,
        cognitive_model: Optional[Callable[..., Dict[str, Any]]] = None,
        *,
        cognitive_params: Optional[Dict[str, float]] = None,
    ) -> "XAIKitTest":
        """Store the cognitive model callable and parameter dictionary."""
        self.cognitive_model = cognitive_model or dummy_cognitive_model
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
        condition_filter: Optional[Dict[str, Any]] = None,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_output_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.output_dir / path

    def _iv_config_for_explanations(
        self,
        methods: Optional[Sequence[Any]],
    ) -> Dict[str, Dict[str, Any]]:
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

    def _require_trials(self) -> List[Dict[str, Any]]:
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

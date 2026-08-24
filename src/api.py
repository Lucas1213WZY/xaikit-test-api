"""High-level XAIKit workflow API."""

from __future__ import annotations

import io
import base64
import html
import json
import uuid
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.virtual_experiment_executor.participant_pools import is_diverse_mode
from src.cognitive_models import (
    default_cognitive_params,
    dummy_cognitive_model,
)
from src.data_loaders import PreparedDataset, prepare_dataset, reencode_prepared_dataset
from src.experiment_planner import (
    DATASET_IV_NAME,
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
    PREDICTION_ONLY_METHOD,
    ensure_prediction_coverage,
)
import src.xai_adapter as xai_adapter_api
import src.cognitive_models as cognitive_models_api
import src.experiment_planner as experiment_planner_api
import src.virtual_experiment_executor as virtual_experiment_api
from src.ai_models import evaluation as ai_eval
from src.experiment_planner import preview as ep_preview
from src.experiment_planner.design_export import (
    DesignExport,
    apply_design_export,
    load_design_export,
    parse_design_export,
)
from src.experiment_planner.protocol import (
    default_study_protocol,
    edit_study_protocol,
    normalize_study_protocol,
    validate_study_protocol,
)


def _model_input_dim(model: Any, _depth: int = 4) -> Optional[int]:
    """Find the feature width a wrapped AI model was built for.

    The wrapper nests the torch module under ``engine`` today and may expose it as
    ``model`` after renaming, so both names are followed rather than hardcoded.
    Returns None when the width cannot be determined (e.g. non-torch models).
    """
    seen = model
    for _ in range(_depth):
        if seen is None:
            return None
        width = getattr(seen, "input_dim", None)
        if isinstance(width, int):
            return width
        seen = getattr(seen, "engine", None) or getattr(seen, "model", None)
    return None


class xaikitTest:
    """Collate one XAI experiment workflow into a single reusable object."""

    def __init__(
        self,
        project_name: str = "xaikit_test",
        *,
        output_dir: str | Path = ".",
        auto_validate_design: bool = True,
        design: Any = None,
    ) -> None:
        """Create a study workflow.

        Args:
            project_name: Study label, used in summaries and output filenames.
            output_dir: Root that relative output paths resolve against.
            auto_validate_design: Re-check the design after each change.
            design: An experiment-design UI export (``DesignExport``, path to the
                JSON, or loaded dict). Registers its IVs, CVs, DVs and protocol
                at once, instead of retyping them as ``add_iv``/``add_cv`` calls.
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.auto_validate_design = auto_validate_design

        self.iv_config, self.CVs, self.DVs = init_experiment_config()

        self.data: Optional[PreparedDataset] = None
        #: Multi-dataset alternative to ``self.data``, keyed by dataset id --
        #: populated by ``prepare_dataset(dataset_id=[...])``. Empty for an
        #: ordinary single-dataset study.
        self.data_by_dataset: dict[str, PreparedDataset] = {}
        #: Per-dataset equivalents of the singular training/explanation fields
        #: below, populated by ``train_AI_model_for_dataset``/
        #: ``explanations_for_dataset`` for a multi-dataset study whose
        #: dataset(s) are not (fully) served by a published corpus -- e.g. one
        #: level covered by CoAX's corpus and another that needs a real
        #: trained model. A dataset served entirely by its corpus has no entry
        #: here; ``use_dataset`` swaps in ``None``/unset for it, which is
        #: correct because ``source="corpus"``/``"assets"`` reads neither.
        self.trained_ai_model_by_dataset: dict[str, Any] = {}
        self.model_manager_by_dataset: dict[str, Any] = {}
        self.model_by_dataset: dict[str, Any] = {}
        self.model_name_by_dataset: dict[str, Any] = {}
        self.training_info_by_dataset: dict[str, Any] = {}
        self.combined_explanations_by_dataset: dict[str, pd.DataFrame] = {}
        self.combined_explanation_path_by_dataset: dict[str, Any] = {}
        self.explanation_paths_by_dataset: dict[str, list[Path]] = {}
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
        #: One row per virtual participant under mode="diverse_participant":
        #: which fitted human it was dealt and the parameters that came with it.
        self.participant_parameters: Optional[pd.DataFrame] = None
        self.participant_parameters_path: Optional[str] = None
        self.study_protocol: dict[str, Any] = default_study_protocol()
        self.walkthrough_previewed: bool = False
        self.walkthrough_approved: bool = False
        self.design_export = None

        if design is not None:
            self.set_design_export(design)

    def set_design_export(self, design: Any, **apply_kwargs: Any) -> "xaikitTest":
        """Register an experiment-design UI export's variables and protocol.

        Args:
            design: A ``DesignExport``, a path to the exported JSON, or the
                already-loaded dict.
            **apply_kwargs: Passed through to ``apply_design_export``:
                ``dv_names`` to restrict which DVs are taken from the export,
                ``include_rvs`` to also register recorded variables, and
                ``show``/``report`` to control the printed summary.
        """
        if isinstance(design, DesignExport):
            parsed = design
        elif isinstance(design, dict):
            parsed = parse_design_export(design)
        else:
            parsed = load_design_export(design)

        self.design_export = parsed
        apply_design_export(self, parsed, **apply_kwargs)
        return self

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
        """Store the researcher-authored, participant-facing study protocol.

        Args:
            study_title: Title shown to participants.
            research_questions: Questions the study asks; list or single string.
            consent_text: Consent wording shown before the study begins.
            procedure_steps: Ordered dicts describing what a participant does.
            study_summary: Short plain-language description of the study.
            start_survey_questions: Questions asked before the trials.
            end_survey_questions: Questions asked after the trials.
            validate: Raise if anything required is missing; False stores a draft.

        Raises:
            ValueError: If ``validate`` is True and the protocol is incomplete.
        """
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
        """Validate and export the study setup to JSON.

        Args:
            path: Destination file, resolved against ``output_dir``.

        Returns:
            The path written.

        Raises:
            ValueError: If the protocol is incomplete.
        """
        problems = validate_study_protocol(self.study_protocol)
        if problems:
            raise ValueError("Study setup is incomplete: " + " ".join(problems))
        output_path = self._resolve_output_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.study_protocol, indent=2), encoding="utf-8")
        return str(output_path)

    def approve_walkthrough(self, *, confirmed: bool = False) -> "xaikitTest":
        """Approve a completed walkthrough, with explicit confirmation required.

        Args:
            confirmed: Must be True, and only after reviewing the walkthrough.

        Raises:
            ValueError: If ``confirmed`` is False or the protocol is incomplete.
            RuntimeError: If the walkthrough has not been previewed.
        """
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
        """Print a concise guide for one workflow stage.

        Args:
            stage: Which stage to explain -- ``design``, ``dataset``,
                ``trial_generation``, ``model_training``,
                ``explanation_generation``, ``cognitive_models`` or
                ``cognitive_simulation``. Common aliases such as ``xai``,
                ``trials`` or ``agents`` are accepted.

        Returns:
            The agent-selection table for ``cognitive_models``, else None.

        Raises:
            ValueError: If ``stage`` is not a known stage or alias.
        """
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
        """Replace the stored IV/CV/DV design dictionaries.

        Args:
            iv_config: Independent variables, keyed by name. Left alone if None.
            cvs: Control variables, keyed by name. Left alone if None.
            dvs: Dependent variables, keyed by name. Left alone if None.
            show: Print the validation summary afterwards.
        """
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
        """Add or replace one independent variable.

        Args:
            name: IV name, e.g. ``xai_method``.
            iv_type: ``within`` (every participant sees every level) or
                ``between`` (each participant sees one level).
            levels: The values to manipulate.
            randomization: For within-subjects IVs, ``block`` or ``trial``.
                Must be omitted for between-subjects IVs.
            show: Print the validation summary afterwards.
        """
        set_iv(self.iv_config, name, iv_type, levels, randomization=randomization)
        if show:
            self.validate_design(show=True)
        return self

    def add_cv(self, name: str, levels: list[Any], *, show: bool = False) -> "xaikitTest":
        """Add or replace one control variable.

        Args:
            name: CV name, e.g. ``user_task``.
            levels: The values held constant or recorded.
            show: Print the validation summary afterwards.
        """
        set_factor(self.CVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def add_dv(self, name: str, levels: list[Any], *, show: bool = False) -> "xaikitTest":
        """Add or replace one dependent variable.

        Args:
            name: DV name, e.g. ``forward_accuracy``.
            levels: The values the measure can take.
            show: Print the validation summary afterwards.
        """
        set_factor(self.DVs, name, levels)
        if show:
            self.validate_design(show=True)
        return self

    def validate_design(self, *, show: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Validate and optionally print the current experimental design.

        Args:
            show: Print the design summary table.

        Returns:
            The validated ``(iv_config, CVs, DVs)`` dictionaries.
        """
        return validate_experiment_config(self.iv_config, self.CVs, self.DVs, show=show)

    def validate(
        self,
        *,
        stage: str = "design",
        strict: bool = False,
        show: bool = True,
    ) -> ValidationReport:
        """Validate this workflow object against the XAIKit support standard.

        Args:
            stage: Which stage to check -- ``design``, ``trial_generation``,
                ``explanation_generation`` or ``execution``. Each stage also
                re-checks the ones before it.
            strict: Treat warnings as failures.
            show: Print the report.

        Returns:
            The report, also stored on ``validation_reports[stage]``.
        """
        report = validate_xaikit_test(
            self,
            stage=stage,
            strict=strict,
            show=show,
        )
        self.validation_reports[report.stage] = report
        return report

    def _prepare_single_dataset(
        self,
        dataset_id: str,
        *,
        model_type: str,
        feature_cols: Optional[Sequence[str]],
        num_features: Optional[int],
        rank_features_by_target: bool,
        use_default_features: bool,
        requires_one_hot_encoding: Optional[bool],
        test_size: float,
        random_state: int,
        show_available: bool,
        show_summary: bool,
        cognitive_model_id: Optional[str],
        custom_dataset: Optional[Any] = None,
    ) -> PreparedDataset:
        if custom_dataset is not None:
            # A caller-supplied TabularDataset (load_custom_dataset), not one
            # of the 9 bundled ids. Feature selection below still applies --
            # in particular the coax/coxam num_features fallback, since
            # coax_loader_feature_cols/coxam_loader_feature_cols will always
            # raise KeyError for a dataset_id they have never seen, correctly
            # routing a custom dataset onto "rank by target correlation, keep
            # the top 5/6" the same way any uncovered bundled dataset does.
            if show_available:
                print(f"Available training datasets: [{dataset_id!r} (custom)]")
            resolved_agent = str(cognitive_model_id or self.cognitive_model_id or "").lower().strip()
            if feature_cols is None and resolved_agent == "coxam" and num_features is None:
                num_features = 6
                rank_features_by_target = True
            elif feature_cols is None and resolved_agent == "coax" and num_features is None:
                num_features = 5
                rank_features_by_target = True
            return prepare_dataset(
                dataset_id,
                dataset=custom_dataset,
                model_type=model_type,
                feature_cols=feature_cols,
                num_features=num_features,
                rank_features_by_target=rank_features_by_target,
                use_default_features=False,
                requires_one_hot_encoding=requires_one_hot_encoding,
                test_size=test_size,
                random_state=random_state,
                show_available=False,
                show_summary=show_summary,
            )

        if str(dataset_id).lower().strip() == "sim2real":
            # No standard raw dataset to split: Sim2Real's rows are a fixed
            # published corpus, and its train/test split is that corpus's own
            # `training`/`test` labels (deltas.csv), not a fresh
            # train_test_split -- feature_cols/num_features/test_size/
            # random_state do not apply and are silently ignored, the same way
            # they would be for any other fixed-corpus source.
            from src.data_loaders import print_dataset_split_summary
            from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
                build_sim2real_dataset_split,
            )

            if show_available:
                print("Available training datasets:", ["sim2real (published corpus)"])
            split = build_sim2real_dataset_split()
            if show_summary:
                print_dataset_split_summary(split)
            return PreparedDataset(split=split)

        resolved_agent = str(cognitive_model_id or self.cognitive_model_id or "").lower().strip()
        if feature_cols is None and use_default_features and resolved_agent == "coxam":
            try:
                from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
                    coxam_loader_feature_cols,
                )

                feature_cols = coxam_loader_feature_cols(dataset_id)
                rank_features_by_target = False
            except KeyError:
                # Not a dataset CoXAM's published corpus covers -- no fixed
                # feature set to route to, so a fresh model gets trained on
                # this dataset instead. CoXAM's own datasets are always
                # 6-feature (COXAM_CORPUS_FEATURES), so match that: rank the
                # full available feature pool by target correlation and keep
                # the top 6, rather than each dataset's own curated default
                # (which can carry a different count, e.g. 5 for
                # wine_quality -- the exact mismatch that made the corpus
                # need its own feature list in the first place).
                if num_features is None:
                    num_features = 6
                rank_features_by_target = True
                use_default_features = False
        elif feature_cols is None and use_default_features and resolved_agent == "coax":
            try:
                from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
                    coax_loader_feature_cols,
                )

                feature_cols = coax_loader_feature_cols(dataset_id)
                rank_features_by_target = False
            except KeyError:
                # Not a dataset CoAX's published corpus covers -- no fixed
                # feature set to route to. CoAX's own datasets are always
                # 5-feature (COAX_CORPUS_FEATURES), so match that shape the
                # same way the CoXAM branch above does.
                if num_features is None:
                    num_features = 5
                rank_features_by_target = True
                use_default_features = False

        return prepare_dataset(
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

    def prepare_dataset(
        self,
        dataset_id: str | Sequence[str],
        *,
        csv_path: Optional[str] = None,
        dataframe: Optional[Any] = None,
        target_col: Optional[str] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        positive_class: Optional[Any] = None,
        threshold: Optional[float] = None,
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
        cognitive_model_id: Optional[str] = None,
    ) -> PreparedDataset | dict[str, PreparedDataset]:
        """Load, optionally feature-select, and split the dataset.

        Args:
            dataset_id: Dataset to load, e.g. ``wine_quality``. Pass a list of
                ids (e.g. ``["wine_quality", "mushrooms"]``) for a
                multi-dataset, between-subjects study: each id is prepared the
                same way and stored in ``self.data_by_dataset`` (keyed by id),
                and a ``dataset`` between-subjects IV with those levels is
                registered automatically -- ``generate_trials()`` then samples
                each participant's trials from only their assigned dataset,
                never mixing instance ids across datasets. When ``csv_path``/
                ``dataframe`` is given, this is just the name recorded on the
                result -- it does not have to be one of the 9 bundled ids.
            csv_path: Load a custom dataset from this CSV instead of one of
                the bundled ids. Pass this or ``dataframe``, not both, and
                pass ``target_col`` alongside either one. Not supported for
                the multi-dataset (list of ids) form.
            dataframe: Load a custom dataset from this in-memory dataframe
                instead of one of the bundled ids. See ``csv_path``.
            target_col: The column to predict, required with ``csv_path``/
                ``dataframe``. Must have exactly two distinct values -- the
                same binary-target restriction every bundled dataset already
                has -- unless ``threshold`` or ``positive_class`` is given.
            categorical_cols: Feature columns to treat as categorical beyond
                the ones auto-detected by dtype (every non-numeric column).
                Only used with ``csv_path``/``dataframe``.
            positive_class: Which ``target_col`` value maps to class 1;
                one-vs-rest for a target with any number of distinct values
                (e.g. Iris' 3-class ``Species``). Ignored when ``threshold``
                is given. Only used with ``csv_path``/``dataframe``.
            threshold: Binarize a continuous/ordinal ``target_col`` as
                ``target_col >= threshold`` -> class 1 -- most real target
                columns are not already two-valued. Only used with
                ``csv_path``/``dataframe``.
            model_type: Model the encoding should suit, e.g. ``mlp``.
            feature_cols: Use exactly these columns, skipping selection.
            num_features: Keep this many features when selecting.
            rank_features_by_target: Rank candidates by association with the
                target before taking ``num_features``.
            use_default_features: Fall back to the dataset's curated default
                feature set when no explicit choice is given.
            requires_one_hot_encoding: Force one-hot encoding on or off;
                None decides from the data.
            test_size: Fraction held out for testing.
            random_state: Seed for the split.
            show_available: Print the datasets XAIKit can load.
            show_summary: Print a summary of the prepared dataset.
            cognitive_model_id: Which agent this dataset is for; defaults to
                ``self.cognitive_model_id``. CoXAM's published corpus was fit
                against its own feature set, which differs from this
                dataset's ordinary default (e.g. wine_quality: 6 features
                including Chlorides, not the loader's usual 5) -- passing
                ``"coxam"`` here, or having already called
                ``set_cognitive_model``/a design export that selected it,
                routes to that set automatically instead of silently
                preparing a dataset ``source="assets"`` cannot use. For a
                dataset the corpus does *not* cover, the top 6 features by
                target correlation are selected instead of that dataset's own
                curated default (which can be a different count) -- CoXAM's
                own datasets are always 6 features, so a freshly trained one
                matches that shape. Has no effect when ``feature_cols`` is
                given explicitly, or for any other agent. A custom dataset
                (``csv_path``/``dataframe``) is never covered by either
                corpus, so it always takes this top-N-by-target-correlation
                path: 5 features for ``"coax"``, 6 for ``"coxam"``.

        Returns:
            The prepared dataset (single id), also stored on ``self.data``; or
            ``{dataset_id: PreparedDataset}`` (list of ids), also stored on
            ``self.data_by_dataset``.
        """
        custom_dataset = None
        if csv_path is not None or dataframe is not None:
            if isinstance(dataset_id, (list, tuple)):
                raise ValueError(
                    "csv_path/dataframe is not supported for a multi-dataset "
                    "study (dataset_id was a list)."
                )
            if target_col is None:
                raise ValueError("target_col is required with csv_path/dataframe.")
            from src.data_loaders import load_custom_dataset

            custom_dataset = load_custom_dataset(
                csv_path=csv_path,
                dataframe=dataframe,
                target_col=target_col,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                positive_class=positive_class,
                threshold=threshold,
                dataset_name=str(dataset_id),
            )

        if isinstance(dataset_id, (list, tuple)):
            dataset_ids = list(dataset_id)
            if len(dataset_ids) < 2:
                raise ValueError(
                    "Pass at least two dataset ids for a multi-dataset study."
                )
            self.data_by_dataset = {
                one_id: self._prepare_single_dataset(
                    one_id,
                    model_type=model_type,
                    feature_cols=feature_cols,
                    num_features=num_features,
                    rank_features_by_target=rank_features_by_target,
                    use_default_features=use_default_features,
                    requires_one_hot_encoding=requires_one_hot_encoding,
                    test_size=test_size,
                    random_state=random_state,
                    show_available=show_available and one_id == dataset_ids[0],
                    show_summary=show_summary,
                    cognitive_model_id=cognitive_model_id,
                )
                for one_id in dataset_ids
            }
            self.data = None
            self.add_iv(DATASET_IV_NAME, "between", dataset_ids)
            return self.data_by_dataset

        self.data = self._prepare_single_dataset(
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
            cognitive_model_id=cognitive_model_id,
            custom_dataset=custom_dataset,
        )
        self.data_by_dataset = {}
        return self.data

    @contextmanager
    def use_dataset(self, dataset_id: str):
        """View one multi-dataset level's own training/explanation state as
        the study's current singular one, so ``evaluate()``/``explanations()``/
        ``_run_agent_experiment()``/the generic executor run unmodified
        against it.

        Read-only: whatever a caller does inside the block against the
        singular fields (``self.data``, ``self.trained_ai_model``, ...) is
        discarded on exit, restoring what was there before -- it does not
        persist back into ``*_by_dataset``. ``train_AI_model_for_dataset``/
        ``explanations_for_dataset`` are the counterparts that do persist,
        since they are the only things meant to change a dataset's stored
        state.
        """
        if dataset_id not in self.data_by_dataset:
            raise KeyError(
                f"No prepared dataset for dataset_id={dataset_id!r}. Prepared: "
                f"{sorted(self.data_by_dataset)}."
            )
        saved = (
            self.data, self.trained_ai_model, self.model_manager, self.model,
            self.model_name, self.training_info, self.metrics,
            self.ai_predictions_by_instance, self.combined_explanations,
            self.combined_explanation_path, self.explanation_paths,
        )
        self.data = self.data_by_dataset[dataset_id]
        self.trained_ai_model = self.trained_ai_model_by_dataset.get(dataset_id)
        self.model_manager = self.model_manager_by_dataset.get(dataset_id)
        self.model = self.model_by_dataset.get(dataset_id)
        self.model_name = self.model_name_by_dataset.get(dataset_id)
        self.training_info = self.training_info_by_dataset.get(dataset_id)
        self.metrics = {}
        self.ai_predictions_by_instance = None
        self.combined_explanations = self.combined_explanations_by_dataset.get(dataset_id)
        self.combined_explanation_path = self.combined_explanation_path_by_dataset.get(dataset_id)
        self.explanation_paths = self.explanation_paths_by_dataset.get(dataset_id, [])
        try:
            yield self
        finally:
            (
                self.data, self.trained_ai_model, self.model_manager, self.model,
                self.model_name, self.training_info, self.metrics,
                self.ai_predictions_by_instance, self.combined_explanations,
                self.combined_explanation_path, self.explanation_paths,
            ) = saved

    def train_AI_model_for_dataset(self, dataset_id: str, **train_kwargs: Any) -> Any:
        """Train a real AI model for one level of a multi-dataset study.

        ``train_AI_model`` operates on ``self.data`` singular -- calling it
        directly in a multi-dataset study would overwrite whichever dataset
        was trained last onto the one shared ``self.trained_ai_model``, and
        the next dataset's training would silently clobber it again. This
        instead trains against ``self.data_by_dataset[dataset_id]``
        specifically and stashes the result into the per-dataset stores, so
        training dataset B never touches dataset A's already-trained model.

        Args:
            dataset_id: Which prepared dataset (from ``data_by_dataset``) to
                train against.
            **train_kwargs: Forwarded to ``train_AI_model`` (``model_type``,
                ``target_score``, ...).

        Returns:
            The trained model, also stored in ``trained_ai_model_by_dataset``.
        """
        if dataset_id not in self.data_by_dataset:
            raise KeyError(
                f"No prepared dataset for dataset_id={dataset_id!r}. Prepared: "
                f"{sorted(self.data_by_dataset)}."
            )
        original_data = self.data
        self.data = self.data_by_dataset[dataset_id]
        try:
            self.train_AI_model(**train_kwargs)
        finally:
            self.trained_ai_model_by_dataset[dataset_id] = self.trained_ai_model
            self.model_manager_by_dataset[dataset_id] = self.model_manager
            self.model_by_dataset[dataset_id] = self.model
            self.model_name_by_dataset[dataset_id] = self.model_name
            self.training_info_by_dataset[dataset_id] = self.training_info
            self.data = original_data
            self.trained_ai_model = None
            self.model_manager = None
            self.model = None
            self.model_name = None
            self.training_info = None
            self.ai_predictions_by_instance = None
        return self.trained_ai_model_by_dataset[dataset_id]

    def explanations_for_dataset(
        self, dataset_id: str, **explanation_kwargs: Any
    ) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
        """Generate one dataset's explanation table, for a multi-dataset study.

        Requires ``train_AI_model_for_dataset(dataset_id, ...)`` to have run
        first -- explanations need a real trained model to explain, and a
        dataset served by a published corpus instead never needs this at all
        (the corpus ships its own explanation vectors).
        """
        if self.trained_ai_model_by_dataset.get(dataset_id) is None:
            raise KeyError(
                f"No trained AI model for dataset_id={dataset_id!r}. Call "
                "train_AI_model_for_dataset(dataset_id, ...) first -- or, if "
                "this dataset is served by a published corpus, no explanation "
                "table is needed for it at all."
            )
        with self.use_dataset(dataset_id):
            path, table = self.explanations(**explanation_kwargs)
            self.combined_explanations_by_dataset[dataset_id] = table
            self.combined_explanation_path_by_dataset[dataset_id] = path
            self.explanation_paths_by_dataset[dataset_id] = list(self.explanation_paths)
        return path, table

    def store_combined_explanations(
        self,
        tables: Sequence[pd.DataFrame],
        *,
        output_dir: str | Path = "generated_explanation",
        model_name: Optional[str] = None,
    ) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
        """Concatenate, save, and store explanation/prediction tables built outside ``explanations()``.

        For workflows that build their own tables directly -- e.g. via
        ``create_xai_method`` for the explanation and a hand-built prediction
        table -- this does the same concat/save/store step ``explanations()``
        does internally, so ``run_experiment()`` and the plotting helpers all
        read from ``combined_explanations`` the same way afterward.

        Args:
            tables: Explanation/prediction DataFrames to concatenate, e.g. a
                prediction-only table and one or more method tables.
            output_dir: Where to save the combined CSV.
            model_name: Used in the saved filename; defaults to ``self.model_name``.

        Returns:
            The saved path and the combined table, also stored on
            ``self.combined_explanations`` / ``self.combined_explanation_path``.
        """
        data = self._require_data()
        config = xai_adapter_api.init_explanation_run(
            data=data,
            iv_config={},
            trained_ai_model=self.trained_ai_model,
            model_name=model_name or self.model_name or "mlp",
            output_dir=self._resolve_output_path(output_dir),
        )
        path, combined = xai_adapter_api.combine_explanation_tables(list(tables), config)
        self.combined_explanation_path = path
        self.combined_explanations = combined
        return path, combined

    def prediction_only_table(self, *, model_name: Optional[str] = None) -> pd.DataFrame:
        """AI predictions for every prepared instance, as prediction-only explanation rows.

        A trial instance needs an AI prediction whether or not it also gets a
        real explanation -- e.g. CoAX's ``none`` condition is still scored
        against it. Pair this with a real method's table (e.g. built directly
        via ``create_xai_method``) and pass both to
        ``store_combined_explanations``.

        Args:
            model_name: Value written to the ``modelName`` column; defaults to
                ``self.model_name``.

        Returns:
            A DataFrame with ``dataId``/``modelName``/``expMethod``/``instanceId``/``pred`` columns.
        """
        data = self._require_data()
        trained_ai_model = self._require_trained_ai_model()
        return xai_adapter_api.prediction_only_table(
            data,
            trained_ai_model,
            model_name=model_name or self.model_name or "mlp",
        )

    def instance_ids_requiring_explanation(self) -> list[int]:
        """Instance ids the generated trials actually need a real explanation for.

        Any trial whose ``xai_method``/``xai_type`` shows one: a training
        trial, or a testing trial with ``tested_w_xai=True``. Use this to
        scope a manually-built explanation (e.g. via ``create_xai_method``) to
        exactly the instances the study's trials will display, instead of
        explaining the whole dataset or guessing a sample size.

        Returns:
            The instance ids, in the order they first appear in the trials.
        """
        return list(self._trial_ids_requiring_explanations() or [])

    def _stamp_resolved_xai_method_on_trials(
        self, resolved_method: str, *, dataset_id: Optional[str] = None
    ) -> None:
        """Give every relevant trial a real ``xai_method`` value.

        ``xai_type`` names the shown explanation family/condition (e.g. CoAX's
        none/importance/attribution), which is not necessarily the same
        vocabulary as the generated method name (CoAX's published corpus
        explains a given dataset with one fixed method regardless of
        xai_type -- see ``COAX_CORPUS_XAI_METHOD``). Without a real
        ``xai_method`` value on each trial, ``get_trial_instance_explanation``
        (the generic executor's lookup, used by baseline models) falls back to
        ``xai_type`` and can never match the pool's real ``expMethod``,
        silently treating every XAI-visible trial as if it had no explanation
        -- a baseline that "succeeds" while reading nothing.

        Args:
            resolved_method: The one method every relevant trial should read
                (a real ``expMethod`` value, e.g. ``"lime"``, not an
                ``xai_type`` like ``"attribution"``).
            dataset_id: Restrict to this dataset's own trial rows -- required
                whenever ``self.trials`` is the *shared* multi-dataset trial
                list (``explanations_for_dataset``/``load_coax_corpus_explanations``
                call this once per dataset while it still is), since stamping
                unscoped would also overwrite another dataset's trials with
                this dataset's resolved method.
        """
        for trial in self.trials:
            if dataset_id is not None and trial.get("dataId") != dataset_id:
                continue
            xai_type = str(trial.get("xai_type", "none")).strip().lower()
            trial["xai_method"] = (
                "none" if xai_type in {"none", "no_xai", "control"} else resolved_method
            )

    def load_coax_corpus_explanations(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        """Load the published CoAX corpus's own explanation vectors as
        ``combined_explanations``, for a declared baseline (KNN, ...) to read
        without needing a real trained AI model.

        A baseline reads predictions/explanations/features -- it does not
        explain the AI model itself -- so a corpus-covered dataset can serve
        it exactly the way ``run_coax_study(source='corpus')`` already serves
        the primary agent, instead of generating a redundant real table.
        Also stamps ``xai_method`` on this dataset's trials (see
        ``_stamp_resolved_xai_method_on_trials``) -- without it the generic
        executor's lookup silently finds nothing, even though the pool itself
        is populated.

        Args:
            dataset_id: Which dataset's corpus rows to load; defaults to
                ``self.data.dataset_id`` for a single-dataset study.

        Returns:
            The loaded table, also stored on ``combined_explanations``.
        """
        from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
            combined_explanations_from_corpus,
        )

        resolved_dataset_id = dataset_id or getattr(self.data, "dataset_id", None)
        if resolved_dataset_id is None:
            raise RuntimeError(
                "No dataset to load corpus explanations for. Call prepare_dataset(...) first."
            )
        pool = combined_explanations_from_corpus(resolved_dataset_id)
        real_methods = pool.loc[pool["expMethod"] != PREDICTION_ONLY_METHOD, "expMethod"]
        if not real_methods.empty and "xai_method" not in self.iv_config:
            self._stamp_resolved_xai_method_on_trials(
                str(real_methods.iloc[0]), dataset_id=resolved_dataset_id
            )
        self.combined_explanations = pool
        return pool

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
        allowed_instance_ids: Optional[Sequence[int]] = None,
        training_instance_ids: Optional[Sequence[int]] = None,
        seed: int = 42,
        output_dir: str | Path = "experiment_output",
        preview_rows: int = 10,
        show: bool = True,
    ) -> TrialGenerationResult:
        """Build training rows followed by held-out testing rows.

        Args:
            model_name: Name recorded on the trials; defaults to the trained model.
            participants_per_between_condition: Simulated participants per
                between-subjects cell.
            num_training: Instances shown in the training phase. In corpus
                mode, drawn from ``training_instance_ids`` instead of a
                prepared dataset's own training split.
            num_testing: Held-out instances shown in the testing phase.
            balance_by_ai_prediction: Draw each phase equally from both predicted
                classes. Requires a trained or loaded AI model; not available
                in corpus mode.
            counterbalancing_strategy: How conditions are ordered across
                participants; ``auto`` picks from the design.
            trial_randomization_strategy: How trials are ordered within a
                participant, e.g. ``balanced``.
            instance_wise_explanation: Give each instance its own explanation
                rather than reusing one per condition.
            shuffle_instances: Shuffle the instance pool before sampling.
            max_trial_instances: Cap on distinct instances explanations are
                generated for.
            allowed_instance_ids: Restrict both phases to instances an external
                corpus can serve -- pass
                ``CoAXAssetRepository.available_instance_ids()`` to keep trials
                runnable against a fixed study set. Unset samples freely.

                **Corpus mode:** if no dataset was prepared (``prepare_dataset()``
                was never called, so ``self.data`` is ``None``) and this is
                given, trials are built directly from these ids instead of a
                prepared dataset's split -- e.g. a Sim2Real study, whose
                stimuli are a fixed published corpus:
                ``allowed_instance_ids=sim2real_available_instance_ids(split="test")``.
                Incompatible with ``balance_by_ai_prediction``. If
                ``num_training > 0`` in this mode, also pass
                ``training_instance_ids``.
            training_instance_ids: Corpus mode's training pool, the
                training-side counterpart to ``allowed_instance_ids`` --
                e.g. ``sim2real_available_instance_ids(split="training")``.
                Required if ``num_training > 0`` in corpus mode; unused
                otherwise.
            seed: Seed for sampling and ordering.
            output_dir: Where the trial tables are written.
            preview_rows: Rows to print when ``show`` is True.
            show: Print a preview of the generated trials.

        Returns:
            The result, also stored on ``trial_result``/``trials``.
        """
        multi_dataset = bool(self.data_by_dataset)
        ai_predictions_by_instance = None
        if multi_dataset:
            if balance_by_ai_prediction:
                raise ValueError(
                    "balance_by_ai_prediction is not yet supported for a "
                    "multi-dataset study (prepare_dataset was given a list of ids)."
                )
            data = None
        elif self.data is None and allowed_instance_ids is not None:
            # Corpus mode: no prepared dataset, trials sample directly from an
            # externally supplied instance pool -- e.g. Sim2Real's published
            # corpus, which prepare_dataset() never touches.
            if balance_by_ai_prediction:
                raise ValueError(
                    "balance_by_ai_prediction needs a trained AI model's "
                    "predictions, which corpus mode (no "
                    "prepare_dataset()/train_AI_model()) never has."
                )
            data = None
        else:
            data = self._require_data()
            if balance_by_ai_prediction:
                if self.ai_predictions_by_instance is not None:
                    ai_predictions_by_instance = self.ai_predictions_by_instance
                elif self.trained_ai_model is not None:
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
                else:
                    raise RuntimeError(
                        "balance_by_ai_prediction=True requires either manually "
                        "supplied predictions (set_ai_predictions(...)) or a "
                        "trained AI model (train_AI_model(...) / "
                        "load_AI_model(...))."
                    )
        if model_name is not None:
            self.model_name = model_name
        self.trial_config = experiment_planner_api.init_trial_build_config(
            data=data,
            data_by_dataset=self.data_by_dataset if multi_dataset else None,
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
            allowed_instance_ids=allowed_instance_ids,
            training_instance_ids=training_instance_ids,
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

    def load_AI_model(
        self,
        *,
        model_type: str = "mlp",
        source: str = "coax",
        dataset_id: Optional[str] = None,
    ) -> Any:
        """Load a pretrained AI model instead of training a new one.

        Weights come from ``assets/model_weights/<model_type>/<source>/``. Use this
        for replications, where predictions and explanations must come from the
        same AI model that produced the published study assets rather than from a
        freshly trained one.

        After this call the study behaves exactly as it would after
        ``train_AI_model``: ``generate_trials(balance_by_ai_prediction=True)`` and
        ``explanations()`` both read from the loaded model.

        Args:
            model_type: Architecture to load, e.g. ``mlp`` or ``xgboost``.
            source: Study the weights came from, e.g. ``coax`` or ``coxam``.
            dataset_id: Dataset the checkpoint was trained on; defaults to the
                prepared dataset.

        Raises:
            ValueError: If the checkpoint's feature width does not match the
                prepared data.
        """
        from src.ai_models import ModelManager

        data = self._require_data()
        dataset_id = dataset_id or data.dataset_id

        self.model_manager = ModelManager()
        self.model = self.model_manager.load_model(
            dataset=dataset_id,
            model_type=model_type,
            source=source,
        )
        # The checkpoint fixes the feature space. Prepared data with a different
        # number of columns would either fail deep inside predict() or, worse,
        # line up by accident against the wrong attributes.
        expected = _model_input_dim(self.model)
        actual = int(data.X_train.shape[1])
        if expected is not None and int(expected) != actual:
            raise ValueError(
                f"Pretrained {model_type} ({source}) for {dataset_id!r} expects "
                f"{int(expected)} features, but the prepared dataset has {actual}: "
                f"{list(getattr(data, 'raw_feature_names', []))}. "
                "Call prepare_dataset(...) with the same feature_cols the weights "
                "were trained on."
            )

        self.model_name = model_type
        self.model_source = source
        self.trained_ai_model = getattr(self.model, "engine", self.model)
        self.training_info = None
        self.training_stdout = ""
        # Any cached predictions came from a different model; drop them.
        self.ai_predictions_by_instance = None
        self.prediction_table_path = None
        self.prediction_table = None
        return self.model

    def set_ai_predictions(self, predictions_by_instance: Mapping[int, Any]) -> None:
        """Manually supply AI predictions keyed by instance id.

        Lets ``generate_trials(balance_by_ai_prediction=True)`` run before
        ``train_AI_model``/``load_AI_model`` -- e.g. when predictions were
        computed elsewhere and trials should be generated first. A later
        ``train_AI_model``/``load_AI_model`` call still resets this, and
        ``explanations()`` still needs a real trained/loaded model.

        Args:
            predictions_by_instance: Mapping from instance id to predicted label.
        """
        self.ai_predictions_by_instance = {
            int(instance_id): _as_python_scalar(prediction)
            for instance_id, prediction in predictions_by_instance.items()
        }

    #: Which named ``train_AI_model`` parameters each model type accepts, split
    #: by where they go: ``build`` reaches the constructor, ``train`` reaches the
    #: training call. Anything named but not listed for the chosen type is a
    #: mistake worth raising on rather than dropping silently.
    _TRAINING_OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
        "mlp": {
            "build": ("hidden_dimension", "dropout_rate", "device_id"),
            "train": ("epochs",),
        },
        "mlp_tf": {
            "build": ("hidden_dimension", "dropout_rate", "device_id"),
            "train": ("epochs",),
        },
        "xgboost": {"build": ("learning_rate", "num_boost_round"), "train": ()},
        "xgboost_tf": {"build": ("learning_rate", "num_boost_round"), "train": ()},
        "sim2real": {"build": ("function_name",), "train": ()},
    }

    def _split_training_options(
        self, model_type: str, **options: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Route the named training parameters to the constructor or the trainer.

        Args:
            model_type: The architecture being trained.
            **options: The named parameters, ``None`` where the caller left them
                unset.

        Returns:
            ``(build_options, train_options)``, each holding only the values the
            caller actually set.

        Raises:
            TypeError: If a set parameter does not belong to ``model_type``.
        """
        allowed = self._TRAINING_OPTIONS.get(model_type)
        if allowed is None:
            known = ", ".join(sorted(self._TRAINING_OPTIONS))
            raise TypeError(f"Unknown model_type {model_type!r}. Use one of: {known}.")

        given = {name: value for name, value in options.items() if value is not None}
        unusable = sorted(set(given) - set(allowed["build"]) - set(allowed["train"]))
        if unusable:
            owners = {
                name: sorted(
                    other
                    for other, groups in self._TRAINING_OPTIONS.items()
                    if name in groups["build"] or name in groups["train"]
                )
                for name in unusable
            }
            detail = "; ".join(f"{name} belongs to {'/'.join(owners[name])}" for name in unusable)
            raise TypeError(
                f"train_AI_model() got {unusable} which model_type={model_type!r} "
                f"cannot use ({detail})."
            )
        return (
            {name: given[name] for name in allowed["build"] if name in given},
            {name: given[name] for name in allowed["train"] if name in given},
        )

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
        epochs: Optional[int] = None,
        hidden_dimension: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        device_id: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_boost_round: Optional[int] = None,
        function_name: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create and train the AI model used for predictions and explanations.

        Every input has its own named parameter. Architecture parameters only
        apply to the model type that owns them, and passing one to a different
        type raises rather than being silently dropped.

        ``model_kwargs`` and ``train_kwargs`` are the older escape hatch and
        still work -- shipped tutorials use ``train_kwargs={"epochs": 100}`` --
        but the named parameters are the supported way, and anything named
        wins over the same key inside a dict.

        Re-encodes the prepared dataset if ``model_type`` needs a different
        one-hot setting than the one it was prepared with.

        Args:
            model_type: Architecture to train -- ``mlp``, ``mlp_tf``,
                ``xgboost``, ``xgboost_tf`` or ``sim2real``.
            source: Study whose defaults to follow, e.g. ``coax`` or ``coxam``.
            target_accuracy: Stop once training accuracy reaches this. Alias for
                ``target_score`` kept for older notebooks.
            target_metric: Metric that ``target_score`` refers to.
            target_score: Stop once ``target_metric`` reaches this; takes
                precedence over ``target_accuracy``. Unset trains a fixed run.
            max_epochs: Upper bound on epochs when training to a target.
            check_every_epochs: How often the target is re-checked.
            batch_size: Training batch size.
            verbose: Let training output print instead of capturing it to
                ``training_stdout``.
            epochs: Training epochs for a fixed run. MLP only.
            hidden_dimension: Width of the MLP's hidden layer. MLP only.
            dropout_rate: MLP dropout. MLP only.
            device_id: CUDA device index, ``-1`` for CPU. MLP only.
            learning_rate: XGBoost learning rate. XGBoost only.
            num_boost_round: XGBoost boosting rounds. XGBoost only.
            function_name: Which analytical function to use. sim2real only.
            model_kwargs: Legacy escape hatch for constructor arguments.
            train_kwargs: Legacy escape hatch for training arguments.

        Returns:
            The trained model, also stored on ``model``/``trained_ai_model``.

        Raises:
            TypeError: If a parameter is given that ``model_type`` cannot use.
        """
        build_options, train_options = self._split_training_options(
            model_type,
            epochs=epochs,
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate,
            device_id=device_id,
            learning_rate=learning_rate,
            num_boost_round=num_boost_round,
            function_name=function_name,
        )
        # Named parameters win over the same key inside the legacy dicts.
        build_options = {**(model_kwargs or {}), **build_options}
        train_options = {**(train_kwargs or {}), **train_options}
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
                **build_options,
            )

            stop_score = target_score if target_score is not None else target_accuracy
            if stop_score is None:
                self.training_info = self.model_manager.train(
                    data.X_train,
                    data.y_train,
                    batch_size=batch_size,
                    **train_options,
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
                    **train_options,
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
        """Plot metric checkpoints from training with a target score.

        Args:
            ax: Existing matplotlib axes to draw on; a new figure is made if None.
        """
        return ai_eval.plot_training_history(self.training_info, ax=ax)

    def predict_labels(self, X: Any, *, threshold: float = 0.5) -> Any:
        """Predicted class labels as ``(n,)`` integers, whatever the model type.

        ``trained_ai_model.predict(X)`` returns each engine's own raw output --
        ``(n, n_classes)`` probabilities from an MLP, ``(n, 2)`` from a binary
        XGBoost objective, ``(n,)`` from the analytical sim2real functions -- so
        calling it directly forces every caller to branch on ``ndim``. This
        resolves that once.

        Args:
            X: Instances to predict.
            threshold: Cutoff for the positive class when the model emits one
                probability per row rather than one per class.

        Raises:
            RuntimeError: If called before ``train_AI_model``.
        """
        return self._require_model_manager().predict_labels(X, threshold=threshold)

    def predict_proba(self, X: Any) -> Any:
        """Class probabilities as ``(n, n_classes)`` floats.

        Raises:
            RuntimeError: If called before ``train_AI_model``.
            ValueError: If the model emits hard labels or continuous values.
        """
        return self._require_model_manager().predict_proba(X)

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
        """Return a labeled confusion matrix for the requested split.

        Args:
            split: ``train`` or ``test``.
            positive_label: Class treated as positive.
            threshold: Probability cutoff for the positive class.
        """
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
        """Plot a labeled confusion matrix for the requested split.

        Args:
            split: ``train`` or ``test``.
            positive_label: Class treated as positive.
            threshold: Probability cutoff for the positive class.
            ax: Existing matplotlib axes to draw on; a new figure is made if None.
        """
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
        """Evaluate the trained AI model with classic classification metrics.

        Args:
            split: ``train``, ``test`` or ``both``.
            positive_label: Class treated as positive.
            threshold: Probability cutoff for the positive class.
            include_report: Also return the per-class precision/recall report.

        Returns:
            Metrics keyed by split, also merged into ``self.metrics``.
        """
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
        """Plot ROC curves and AUC values for train/test predictions.

        Args:
            split: ``train``, ``test`` or ``both``.
            positive_label: Class treated as positive.
            ax: Existing matplotlib axes to draw on; a new figure is made if None.
        """
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
        """Generate method-level XAI tables and combine them into one table.

        Args:
            methods: XAI methods to run. Defaults to the levels of the stored
                ``xai_method``/``xai_type`` IV.
            model_name: Name used in explanation filenames; defaults to the
                trained model's.
            output_dir: Where the per-method tables are written.
            target: Class index explanations are generated for.
            method_kwargs: Per-method options, keyed by method name, e.g.
                ``{"shap": {"background_size": 100}}``.
            show_checks: Print the resolved methods and validation output.

        Returns:
            Path to the combined table and the table itself.

        Raises:
            RuntimeError: If no methods are given and none are stored.
        """
        # A design that names an explanation family (xai_type) or a property
        # (xai_property) but no xai_method is complete -- infer the methods the
        # framework generates rather than leaving the run with nothing to do.
        if methods is None and "xai_method" not in self.iv_config:
            inferred = self._inferred_xai_methods()
            if inferred:
                methods = inferred
                if show_checks:
                    print(
                        f"No xai_method IV; using {self._resolved_framework()}'s methods: "
                        f"{inferred}"
                    )

        explanation_iv_config = self._iv_config_for_explanations(methods)
        resolved_methods = self._xai_methods_from_iv_config(explanation_iv_config)
        if not resolved_methods:
            raise RuntimeError(
                "No XAI methods were provided and no `xai_method`/`xai_type` IV is stored. "
                "Call `add_iv('xai_method', ..., [...])` or pass `methods=[...]`."
            )

        if "xai_method" not in self.iv_config and len(resolved_methods) == 1 and self.trials:
            self._stamp_resolved_xai_method_on_trials(str(resolved_methods[0]))

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
        """Visualize one local explanation using the XAI adapter plot helper.

        Args:
            visualization: Plot style, e.g. ``influence`` or ``importance``.
            method: XAI method to show; None picks the first with an explanation.
            instance_id: Instance to show; None picks the first explained one.
            top_n: Number of features to display.
            class_labels: Display names for the classes.
            phase: Trial phase being previewed; ``testing`` hides the AI
                prediction by default.
            show_ai_prediction: Force the AI prediction on or off, overriding
                what ``phase`` implies.
            **kwargs: Passed through to the underlying plot helper.

        Raises:
            ValueError: If no generated explanation matches the request.
        """
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
        """Interactively preview one participant's trials with Back/Next controls.

        Args:
            participant_id: Which participant's sequence to walk through.
            visualization: Plot style used for each explanation.
            top_n: Number of features to display.
            class_labels: Display names for the classes.
            fallback: What to show when a trial has no explanation; ``auto``
                substitutes the prediction-only view.
        """
        data = self._require_data()
        trials = self._require_trials()
        pool = ensure_prediction_coverage(
            self._require_combined_explanations(),
            trials=trials,
            data=data,
            trained_ai_model=self.trained_ai_model,
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
        """Preview the full participant journey and expose final approval controls.

        Marks the walkthrough as previewed, which ``approve_walkthrough``
        requires.

        Args:
            participant_id: Which participant's journey to walk through.
            explanation_pool: Explanations to draw from; defaults to the
                combined table.
            visualization: Plot style used for each explanation.
            top_n: Number of features to display.
            class_labels: Display names for the classes.
            max_trials: Stop after this many trials; None shows all.
            fallback: What to show when a trial has no explanation.
        """
        data = self._require_data()
        trials = self._require_trials()
        pool = explanation_pool if explanation_pool is not None else self._require_combined_explanations()
        pool = ensure_prediction_coverage(
            pool,
            trials=trials,
            data=data,
            trained_ai_model=self.trained_ai_model,
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
        """Store the cognitive model callable and parameter dictionary.

        Args:
            cognitive_model: A callable standing in for a participant. Omit when
                selecting a built-in agent by id.
            cognitive_model_id: Built-in agent to use -- a machine-proxy baseline
                (``knn``, ``decision_tree``, ``logistic_regression``, ``mlp``) or
                a cognitive agent (``coax``, ``coxam``, ``sim2real``).
            cognitive_params: Parameters for a cognitive agent, e.g.
                ``retrieval_threshold``. Defaults apply when omitted.
            model_kwargs: Constructor arguments for a machine-proxy baseline.
                Only valid alongside ``cognitive_model_id``.

        Raises:
            ValueError: If ``model_kwargs`` is combined with an explicit
                ``cognitive_model``.
        """
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
            ({} if is_baseline else default_cognitive_params(self.cognitive_model_id))
            if cognitive_params is None
            else deepcopy(cognitive_params)
        )
        return self

    #: Agents whose simulation lives in a dedicated runner rather than in the
    #: generic executor. ``set_cognitive_model`` leaves ``cognitive_model`` as
    #: the placeholder for these, because the runner owns the model.
    _AGENT_RUNNERS = ("coax", "coxam", "sim2real")

    #: What each agent's runner calls its flat cognitive-parameter argument.
    #: CoAX is absent on purpose -- its parameters are nested per condition
    #: (``{(xai_type, tested): {...}}``) rather than one flat mapping, so a flat
    #: ``cognitive_params`` cannot be routed there without inventing a key.
    _COGNITIVE_PARAM_KEYWORDS = {"coxam": "eval_params", "sim2real": "cognitive_params"}

    def _run_agent_experiment(
        self,
        *,
        mode: str,
        participant_id: Optional[int],
        condition_filter: Optional[dict[str, Any]],
        explanation_pool: Optional[pd.DataFrame],
        runner_kwargs: dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Dispatch to a research agent's runner, or return None for the generic path.

        Kept separate from ``run_experiment`` so the generic executor's own
        preparation -- ``ensure_prediction_coverage`` in particular -- is not
        run for agents that do not use it.
        """
        agent = (self.cognitive_model_id or "").lower().strip()
        if agent not in self._AGENT_RUNNERS:
            if runner_kwargs:
                raise TypeError(
                    f"run_experiment() got unexpected keyword argument(s) "
                    f"{sorted(runner_kwargs)!r}. These are runner options for "
                    f"{' or '.join(self._AGENT_RUNNERS)}; the current cognitive model is "
                    f"{self.cognitive_model_id!r}."
                )
            return None

        self._require_data()
        self._require_trials()

        # Carry the stored cognitive parameters into the runner. Without this
        # `set_cognitive_model(cognitive_params=...)` is parsed and dropped: the
        # run succeeds using the agent's defaults, so a UI-configured parameter
        # silently has no effect. Each runner names the argument differently,
        # and an explicit runner_kwargs value still wins.
        parameter_keyword = self._COGNITIVE_PARAM_KEYWORDS.get(agent)
        if parameter_keyword and self.cognitive_params and parameter_keyword not in runner_kwargs:
            runner_kwargs[parameter_keyword] = dict(self.cognitive_params)

        if agent == "sim2real":
            from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_study_runner import (
                run_sim2real_study,
            )

            # No explanation_pool and no trained model: Sim2Real's stimuli,
            # explanations and counterfactual ground truth are a fixed corpus.
            self.simulated_results = run_sim2real_study(
                self,
                mode=mode,
                participant_id=participant_id,
                condition_filter=condition_filter,
                store=True,
                **runner_kwargs,
            )
            return self.simulated_results

        if agent == "coxam":
            from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
                run_coxam_study,
            )

            # No explanation_pool: coxam fits its own DT/LR surrogates, because
            # the combined table carries the generic a{i}_i attribution schema
            # rather than the coef_a{i}/tree_structure its interpreters need.
            self.simulated_results = run_coxam_study(
                self,
                mode=mode,
                participant_id=participant_id,
                condition_filter=condition_filter,
                store=True,
                **runner_kwargs,
            )
            return self.simulated_results

        from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
            run_coax_study,
        )

        self.simulated_results = run_coax_study(
            self,
            mode=mode,
            participant_id=participant_id,
            condition_filter=condition_filter,
            explanation_pool=explanation_pool,
            store=True,
            **runner_kwargs,
        )
        return self.simulated_results

    def _run_multi_dataset_experiment(
        self,
        *,
        mode: str,
        participant_id: Optional[int],
        condition_filter: Optional[dict[str, Any]],
        explanation_pool: Optional[pd.DataFrame],
        runner_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """Run each dataset's own single-dataset runner and concatenate.

        ``prepare_dataset(dataset_id=[...])`` never merges instance-id spaces --
        one dataset's instance ids mean nothing in another's feature space -- so
        this does not rewire the agent runners to be dataset-aware internally.
        Instead it loops the *unmodified* runner once per dataset, temporarily
        pointing ``self.data``/``self.trials`` at that dataset's own prepared
        data and trial rows (already tagged with the matching ``dataId`` by
        ``generate_trials()``), then restores the multi-dataset state.
        """
        all_trials = self._require_trials()
        levels = list(self.data_by_dataset)
        trials_by_level = {
            level: [trial for trial in all_trials if trial.get("dataId") == level]
            for level in levels
        }
        missing = [level for level, rows in trials_by_level.items() if not rows]
        if missing:
            raise RuntimeError(
                f"No trial rows found for dataset(s) {missing!r}. "
                "Call generate_trials() on this study first."
            )

        original_trials = self.trials
        results: list[pd.DataFrame] = []
        drawn_parameters: list[pd.DataFrame] = []
        self.participant_parameters = None
        try:
            for level_index, level in enumerate(levels):
                self.trials = trials_by_level[level]
                level_kwargs = dict(runner_kwargs)
                # Each level runs its own draw; without a per-level offset every
                # level would deal its pool rows in the same order. The levels'
                # participants are disjoint by construction, so the assignments
                # stay independent.
                if is_diverse_mode(mode) and "sampling_seed" not in level_kwargs:
                    level_kwargs["sampling_seed"] = level_index
                # A dataset this study trained a real model for (see
                # train_AI_model_for_dataset) needs source="study"/"fit"; one
                # served entirely by a published corpus needs "corpus"/
                # "assets". Mixed multi-dataset studies need this decided per
                # level, so the server passes a {level: source} mapping here
                # instead of one flat string when it knows the mix -- a plain
                # string (or nothing) is left alone, same as before.
                source = level_kwargs.get("source")
                if isinstance(source, dict):
                    level_kwargs["source"] = source.get(level)
                with self.use_dataset(level):
                    agent_results = self._run_agent_experiment(
                        mode=mode,
                        participant_id=participant_id,
                        condition_filter=condition_filter,
                        explanation_pool=explanation_pool,
                        runner_kwargs=level_kwargs,
                    )
                if agent_results is None:
                    raise ValueError(
                        "Multi-dataset simulation requires a research-agent "
                        f"cognitive model (one of {sorted(self._AGENT_RUNNERS)}); "
                        f"{self.cognitive_model_id!r} is not one."
                    )
                if agent_results.empty:
                    # This level had no rows matching participant_id/condition_filter
                    # -- expected for participant_by_participant, since participant
                    # ids are disjoint across dataset levels.
                    continue
                tagged = agent_results.copy()
                tagged[DATASET_IV_NAME] = level
                results.append(tagged)
                if self.participant_parameters is not None:
                    level_parameters = self.participant_parameters.copy()
                    level_parameters[DATASET_IV_NAME] = level
                    drawn_parameters.append(level_parameters)
                    self.participant_parameters = None
        finally:
            self.trials = original_trials

        if not results:
            raise RuntimeError(
                f"No trials matched mode={mode!r}, participant_id={participant_id!r}, "
                f"condition_filter={condition_filter!r} in any dataset."
            )
        self.simulated_results = pd.concat(results, ignore_index=True)
        if drawn_parameters:
            self.participant_parameters = pd.concat(drawn_parameters, ignore_index=True)
        return self.simulated_results

    def run_experiment(
        self,
        *,
        mode: str = "participant_by_participant",
        participant_id: Optional[int] = 1,
        condition_filter: Optional[dict[str, Any]] = None,
        explanation_pool: Optional[pd.DataFrame] = None,
        require_walkthrough_approval: bool = False,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        """Run the cognitive simulation over selected generated trial rows.

        Which simulation runs is decided by ``cognitive_model_id``. The three
        research agents have their own runners and are dispatched to here, so
        that selecting one and calling this method actually runs it:

        * ``coax``     -> :func:`run_coax_study`
        * ``coxam``    -> :func:`run_coxam_study`
        * ``sim2real`` -> :func:`run_sim2real_study`

        Everything else -- the baseline models and any explicitly supplied
        ``cognitive_model`` -- runs through the generic executor as before.

        Args:
            mode: ``participant_by_participant`` runs one participant;
                other modes run the whole set. ``diverse_participant`` runs the
                whole set *and* deals every participant its own cognitive
                parameters, drawn from the humans the agent was fitted to and
                filtered to that participant's own condition -- see below.
            participant_id: Which participant to run in per-participant mode.
            condition_filter: Restrict to trials matching these IV values.
            explanation_pool: Explanations to draw from; defaults to the
                combined table. Ignored by ``coxam``, which fits its own
                surrogates rather than reading the combined table.
            require_walkthrough_approval: Refuse to run until the walkthrough has
                been previewed and approved.
            **runner_kwargs: Passed to the selected agent's runner, e.g.
                ``user_task="counterfactual_simulation"``, ``source=``,
                ``policy_override=`` or ``eval_params=`` for coxam, and
                ``cognitive_model=`` or ``train_with_explanation=`` for coax.
                Rejected for the baseline path, which takes no such options.

        Returns:
            Simulated responses, also stored on ``simulated_results``.

        Raises:
            RuntimeError: If approval is required but has not been given.
            TypeError: If ``runner_kwargs`` is passed without an agent selected.
            ValueError: If ``diverse_participant`` is asked of a model with no
                fitted human population behind it.

        ``mode="diverse_participant"`` exists because every other mode runs one
        parameter set for the whole study, which makes each condition's N
        participants one person taking N sessions. The effect is not subtle: a
        Sim2Real study reports a within-condition SD of exactly 0.0 in three of
        four conditions (t reaches 1e15 and p underflows to zero), and a CoXAM
        study reports p = 2.7e-10 for an effect whose only variance source is
        which instances each participant happened to see. In diverse mode each
        participant is dealt one real fitted participant's parameters from
        ``assets/human_data``, and the assignment is recorded on every result
        row and in ``participant_parameters``. Tune it with ``sampling_seed``,
        ``sampling_replace`` and ``parameter_pool``.
        """
        if is_diverse_mode(mode) and (self.cognitive_model_id or "") not in self._AGENT_RUNNERS:
            raise ValueError(
                "mode='diverse_participant' draws parameters from the humans an "
                f"agent was fitted to, and {self.cognitive_model_id!r} has no such "
                "population. Select a research agent "
                f"({', '.join(self._AGENT_RUNNERS)}) via set_cognitive_model"
                "(cognitive_model_id=...), or run mode='whole_experiment'."
            )
        if require_walkthrough_approval and not self.walkthrough_approved:
            raise RuntimeError(
                "Experiment execution is locked. Preview the complete walkthrough and "
                "click 'Approve walkthrough' first."
            )
        if not is_diverse_mode(mode):
            # Otherwise a later shared-parameter run would still save (and claim)
            # the previous diverse run's assignment.
            self.participant_parameters = None
            self.participant_parameters_path = None

        if self.data_by_dataset:
            return self._run_multi_dataset_experiment(
                mode=mode,
                participant_id=participant_id,
                condition_filter=condition_filter,
                explanation_pool=explanation_pool,
                runner_kwargs=runner_kwargs,
            )

        agent_results = self._run_agent_experiment(
            mode=mode,
            participant_id=participant_id,
            condition_filter=condition_filter,
            explanation_pool=explanation_pool,
            runner_kwargs=runner_kwargs,
        )
        if agent_results is not None:
            return agent_results

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
            # Not _require_trained_ai_model(): a corpus-covered study (CoAX/
            # CoXAM) ships its own predictions for every trial instance, so no
            # model is needed. ensure_prediction_coverage demands one only if a
            # genuine gap has to be predicted.
            trained_ai_model=self.trained_ai_model,
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

    def condition_counts(
        self,
        results: Optional[pd.DataFrame] = None,
        *,
        by: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Row counts per condition, to sanity-check trial/response coverage.

        Answers "did every condition get the rows I expect" -- the usual
        check right after ``generate_trials()`` or ``run_experiment()``.

        Args:
            results: Rows to count; defaults to ``self.simulated_results``,
                falling back to the generated trials if no simulation has run
                yet.
            by: Columns to group by; defaults to ``phase`` plus every
                registered IV name that is actually present in ``results``.

        Returns:
            One row per group, with a ``rows`` count column.
        """
        if results is None:
            results = (
                self.simulated_results
                if self.simulated_results is not None
                else pd.DataFrame(self.trials)
            )
        if by is None:
            by = [column for column in ("phase", *self.iv_config) if column in results.columns]
        return results.groupby(list(by), dropna=False).size().reset_index(name="rows")

    def save_results(
        self,
        *,
        out_dir: str | Path = "experiment_output",
    ) -> tuple[str, str]:
        """Save simulated experiment results as CSV and JSON.

        Args:
            out_dir: Directory the files are written to.

        Returns:
            The CSV and JSON paths. A ``diverse_participant`` run also writes
            ``participant_parameters.csv`` next to them and records it on
            ``participant_parameters_path``; the return value stays a 2-tuple.

        Raises:
            RuntimeError: If called before ``run_experiment``.
        """
        if self.simulated_results is None:
            raise RuntimeError("Call run_experiment(...) before save_results(...).")
        resolved_out_dir = self._resolve_output_path(out_dir)
        self.simulated_csv_path, self.simulated_json_path = virtual_experiment_api.save_simulated_results(
            self.simulated_results,
            out_dir=resolved_out_dir,
        )
        # A side artifact, deliberately not a third return value: the server
        # unpacks exactly two paths from this call.
        if self.participant_parameters is not None:
            path = Path(resolved_out_dir) / "participant_parameters.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.participant_parameters.to_csv(path, index=False)
            self.participant_parameters_path = str(path)
        return self.simulated_csv_path, self.simulated_json_path

    def analyze_iv_dv(
        self,
        *,
        iv: str,
        dv: str,
        participant_column: str = "participantId",
    ) -> Any:
        """Analyze one stored DV against one IV using testing responses.

        Args:
            iv: Independent variable to group by.
            dv: Dependent variable to test.
            participant_column: Column identifying participants, used to
                aggregate within participant before testing.

        Raises:
            RuntimeError: If called before ``run_experiment``.
        """
        if self.simulated_results is None:
            raise RuntimeError("Call run_experiment(...) before analyze_iv_dv(...).")
        from src.statistical_analyst import analyze_iv_dv

        return analyze_iv_dv(
            self.simulated_results,
            iv=iv,
            dv=dv,
            participant_column=participant_column,
        )

    def pairwise_condition_tests(
        self,
        *,
        dv: str,
        condition_cols: Sequence[str],
        participant_column: str = "participantId",
        correction: Optional[str] = "holm",
        phase: str = "testing",
    ) -> Any:
        """Pairwise t-tests (paired where participants overlap) between every
        crossed condition cell, e.g. xai_type x tested_w_xai.

        Args:
            dv: Dependent variable to compare.
            condition_cols: Columns whose crossed values define the cells
                being compared, e.g. ``["xai_type", "tested_w_xai"]``.
            participant_column: Column identifying participants.
            correction: Multiple-comparison correction (``"holm"`` etc.);
                None reports raw p-values only.
            phase: Restrict to one trial phase before comparing.

        Raises:
            RuntimeError: If called before ``run_experiment``.
        """
        if self.simulated_results is None:
            raise RuntimeError("Call run_experiment(...) before pairwise_condition_tests(...).")
        from src.statistical_analyst import pairwise_condition_tests

        data = self.simulated_results
        if phase and "phase" in data:
            data = data[data["phase"].astype(str).str.lower() == phase.lower()]
        return pairwise_condition_tests(
            data,
            value_col=dv,
            condition_cols=condition_cols,
            participant_col=participant_column,
            correction=correction,
        )

    def plot_results_grid(
        self,
        *,
        responses: Optional[pd.DataFrame] = None,
        ivs: Optional[Sequence[str]] = None,
        dvs: Optional[Sequence[str]] = None,
        participant_column: str = "participantId",
        phase: Optional[str] = "testing",
        errorbar: Optional[str] = "ci95",
        title: Optional[str] = "Experiment results",
        value_labels: bool = True,
    ) -> Any:
        """Plot every requested dependent variable against every requested IV.

        Args:
            responses: Results to plot; defaults to the stored simulation.
            ivs: IVs to plot; defaults to every IV in the design.
            dvs: DVs to plot; defaults to every DV in the design.
            participant_column: Column identifying participants.
            phase: Restrict to one trial phase, e.g. ``testing``.
            errorbar: Error bar statistic -- ``ci95`` (Student-t 95% CI half-width,
                the default), ``sem``, ``std``, or None to hide them.
            title: Figure title.
            value_labels: Print the value above each bar.

        Raises:
            RuntimeError: If called before ``run_experiment``.
        """
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
        errorbar: Optional[str] = "ci95",
        x_levels: Optional[Sequence[Any]] = None,
        hue_levels: Optional[Sequence[Any]] = None,
        x_labels: Optional[dict[Any, str]] = None,
        hue_labels: Optional[dict[Any, str]] = None,
        title: Optional[str] = None,
        value_labels: bool = True,
    ) -> Any:
        """Plot one DV against two IVs as grouped participant-level means.

        Args:
            x_iv: IV placed on the x axis.
            hue_iv: IV distinguished by colour within each x group.
            dv: Dependent variable to plot.
            responses: Results to plot; defaults to the stored simulation.
            participant_column: Column identifying participants.
            phase: Restrict to one trial phase, e.g. ``testing``.
            errorbar: Error bar statistic -- ``ci95`` (Student-t 95% CI half-width,
                the default), ``sem``, ``std``, or None to hide them.
            x_levels: Order of x-axis levels; defaults to the design's order.
            hue_levels: Order of colour levels; defaults to the design's order.
            x_labels: Display names for x-axis levels.
            hue_labels: Display names for colour levels.
            title: Figure title.
            value_labels: Print the value above each bar.

        Raises:
            RuntimeError: If called before ``run_experiment``.
        """
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

    def compare_to_human_data(
        self,
        human_responses: pd.DataFrame,
        *,
        group_column: str,
        human_column: str,
        model_column: str,
        responses: Optional[pd.DataFrame] = None,
        on: Optional[Sequence[str]] = None,
        participant_column: str = "participantId",
        model_name: Optional[str] = None,
        dv: str = "Agreement with the AI",
        title: Optional[str] = None,
        note: str = "",
        order: Optional[Sequence[Any]] = None,
        group_labels: Optional[dict[Any, str]] = None,
    ) -> Any:
        """Compare the simulation against the humans it is meant to reproduce.

        This is the evidence step after ``run_experiment``: one grouped-bar panel
        per grouping level, human beside cognitive model. Bars are the mean over
        *participant* means -- never over trials -- so a participant with more
        trials does not count for more, and the error bar is the SEM of those
        participant means.

        ``human_responses`` and the simulation are joined on ``on`` when given;
        pass a single already-aligned frame as ``human_responses`` and leave
        ``on=None`` to skip the join, which is what the published fit tables
        (already carrying both a human and a model column) need.

        Args:
            human_responses: Human trials, or an already-aligned frame.
            group_column: Column whose values become the category axis.
            human_column: Column holding the human measure.
            model_column: Column holding the cognitive model's measure.
            responses: Simulation to compare; defaults to the stored one. Unused
                when ``on`` is None.
            on: Columns to join the human and simulated frames on. None means
                ``human_responses`` already carries both measures.
            participant_column: Column identifying participants.
            model_name: Series label for the model; defaults to
                ``cognitive_model_id`` or ``"Cognitive model"``.
            dv: What the numbers measure; becomes the y-axis label.
            title: Panel heading; defaults to naming the grouping column.
            note: Caveat shown under the heading.
            order: Category order; defaults to sorted.
            group_labels: Display labels for category values.

        Returns:
            A :class:`~src.result_visualizer.ComparisonPanel`. Call
            ``.to_frame()`` for the numbers, or pass it to
            ``render_human_vs_model_report`` for the interactive page.

        Raises:
            RuntimeError: If ``on`` is given and no simulation is available.
        """
        from src.result_visualizer import comparison_panel

        frame = human_responses
        if on is not None:
            simulated = responses if responses is not None else self.simulated_results
            if simulated is None:
                raise RuntimeError(
                    "Call run_experiment(...) before compare_to_human_data(..., on=...), "
                    "or pass an already-aligned frame with on=None."
                )
            frame = human_responses.merge(simulated, on=list(on), how="inner")
            if frame.empty:
                raise ValueError(
                    f"Joining human responses to the simulation on {list(on)} matched no rows."
                )

        label = model_name or (self.cognitive_model_id or "Cognitive model")
        return comparison_panel(
            frame,
            participant_column=participant_column,
            group_column=group_column,
            series={"Human": human_column, str(label): model_column},
            title=title or f"Human vs {label}, by {group_column}",
            dv=dv,
            note=note,
            order=order,
            group_labels=group_labels,
        )

    def render_human_vs_model_report(
        self,
        panels: Any,
        path: str | Path = "human_vs_model_report.html",
        *,
        study_name: Optional[str] = None,
        task: str = "",
        participants: Optional[int] = None,
        title: str = "Human vs cognitive model",
    ) -> Path:
        """Write comparison panels to one self-contained interactive HTML page.

        Args:
            panels: One panel from :meth:`compare_to_human_data`, a list of them,
                or a list of ``ComparisonStudy`` to render several studies.
            path: Output path; relative paths land in ``output_dir``.
            study_name: Study heading; defaults to ``cognitive_model_id``.
            task: Sub-heading describing what participants were asked.
            participants: Participant count shown beside the heading.
            title: Page title.

        Returns:
            The path written.
        """
        from src.result_visualizer import (
            ComparisonPanel,
            ComparisonStudy,
            render_comparison_report,
        )

        if isinstance(panels, ComparisonPanel):
            panels = [panels]
        if all(isinstance(item, ComparisonStudy) for item in panels):
            studies = list(panels)
        else:
            studies = [
                ComparisonStudy(
                    name=study_name or (self.cognitive_model_id or "Cognitive model"),
                    task=task,
                    participants=participants,
                    panels=list(panels),
                )
            ]
        return render_comparison_report(
            studies, self._resolve_output_path(path), title=title
        )

    def _resolve_output_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.output_dir / path

    #: Which XAI methods generate each framework's explanations, when the design
    #: names an explanation family but no ``xai_method`` IV. A design that does
    #: declare ``xai_method`` always wins over these.
    XAI_METHODS_BY_FRAMEWORK: dict[str, tuple[str, ...]] = {
        "coax": ("lime", "shap"),
        # Sim2Real's corpus is built from LIME attributions; the study varies
        # which property the explanation was optimized for, not the method.
        "sim2real": ("lime",),
        # CoXAM's xai_type names the surrogate family directly, so the method
        # follows from the level -- see _COXAM_METHODS_BY_XAI_TYPE.
        "coxam": ("decision_tree", "logistic_regression"),
    }

    #: A CoXAM ``xai_type`` level and the surrogate(s) it displays. ``hybrid``
    #: shows both families, so it needs both generated.
    _COXAM_METHODS_BY_XAI_TYPE: dict[str, tuple[str, ...]] = {
        "decision_tree": ("decision_tree",),
        "logistic_regression": ("logistic_regression",),
        "hybrid": ("decision_tree", "logistic_regression"),
    }

    def _resolved_framework(self) -> str:
        """The agent this design runs, inferring Sim2Real from its IV.

        Mirrors ``DesignExport.resolved_framework``: a design that varies
        ``xai_property`` is a Sim2Real study even when the cognitive model id
        says ``coax``, because Sim2Real's model is CoAX-derived.
        """
        if "xai_property" in self.iv_config:
            return "sim2real"
        return str(self.cognitive_model_id or "").lower().strip()

    def _methods_for_explanation_family(self, family: Any) -> tuple[str, ...]:
        """The XAI method(s) that generate one explanation-family level.

        For CoXAM the level names the family, so ``hybrid`` needs both
        surrogates. Every other framework generates the same methods whatever
        the level, so the level only decides *whether* an explanation is shown.
        """
        framework = self._resolved_framework()
        level = str(family).lower().strip().replace(" ", "_").replace("-", "_")
        if level in {"none", "no_xai", "control"}:
            return ()
        if framework == "coxam":
            return self._COXAM_METHODS_BY_XAI_TYPE.get(level, ())
        return self.XAI_METHODS_BY_FRAMEWORK.get(framework, ())

    def _inferred_xai_methods(self) -> list[str]:
        """XAI methods this design needs, when it declares no ``xai_method`` IV.

        A design can name the explanation *family* it shows (``xai_type``) or
        the *property* it varies (``xai_property``) without naming a method --
        both are complete designs. Previously that combination generated no
        explanations at all and raised nothing, because the per-method instance
        map was keyed by family while generation ran by method name.
        """
        framework = self._resolved_framework()
        if framework == "coxam":
            levels = self.iv_config.get("xai_type", {}).get("levels", [])
            methods: list[str] = []
            for level in levels:
                for method in self._methods_for_explanation_family(level):
                    if method not in methods:
                        methods.append(method)
            return methods
        if framework == "coax":
            # Unlike CoXAM, a CoAX xai_type level (none/importance/attribution)
            # does not decide the method -- the published corpus explains a
            # given dataset with exactly one method regardless of xai_type
            # (see COAX_CORPUS_XAI_METHOD), so a study resolves to that one
            # method rather than generating every method CoAX has ever used.
            # ``self.data`` (not ``data_by_dataset``) is the source of truth
            # for which single dataset applies: a plain single-dataset study
            # sets it directly, and ``explanations_for_dataset``'s
            # ``use_dataset`` swap sets it just as unambiguously for one level
            # of a multi-dataset study -- ``data_by_dataset`` being populated
            # at the same time does not make the dataset any less known.
            from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
                COAX_CORPUS_XAI_METHOD,
            )

            data = getattr(self, "data", None)
            if data is not None:
                method = COAX_CORPUS_XAI_METHOD.get(data.dataset_id)
                if method:
                    return [method]
                # No corpus precedent for this dataset (a freshly-trained one,
                # e.g. mushrooms) -- CoAX still only ever needs one explanation
                # vector per instance (attribution/importance are the same
                # vector, abs-transformed), so generating every framework
                # method here would leave trials with no xai_method column to
                # disambiguate between them at simulation time. Default to the
                # framework's first method rather than all of them.
                default_methods = self.XAI_METHODS_BY_FRAMEWORK.get(framework, ())
                if default_methods:
                    return [default_methods[0]]
        return list(self.XAI_METHODS_BY_FRAMEWORK.get(framework, ()))

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
        by_key = {
            key: list(dict.fromkeys(group["instanceId"].astype(int).tolist()))
            for key, group in required.groupby("_method_key", sort=False)
        }
        if method_column == "xai_method":
            return by_key

        # The trials carry an explanation *family*, not a method name, so the
        # keys have to be translated -- generation runs by method, and a map
        # keyed "attribution" would leave method "shap" with no instances and
        # silently produce no explanations at all.
        by_method: dict[str, list[int]] = {}
        for family, instance_ids in by_key.items():
            for method in self._methods_for_explanation_family(family):
                merged = by_method.setdefault(method, [])
                merged.extend(
                    instance_id for instance_id in instance_ids if instance_id not in merged
                )
        return by_method or by_key

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

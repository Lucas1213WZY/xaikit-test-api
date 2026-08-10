"""CoXAM experiment execution helpers built on the current data-loader API.

Mirrors ``coax_trial_executor.py``'s role for the CoAX integration, adapted to
CoXAM's shape: a trained RL meta-policy (``MaskablePPO``) choosing between
three trained sub-policies, driving a ``gymnasium`` environment
(``CombinedStrategyPolicyEnv``) that tracks per-episode history state, rather
than a stateless per-trial strategy class.

Nothing in ``src/cognitive_models/cognitive_models/coxam/`` is modified here
beyond the import-wiring fix already applied to make it importable from its
new package location. ``TrialDrivenCombinedStrategyPolicyEnv`` below is a
subclass that overrides two hook methods the base environment already calls
through ``self.<method>(...)`` -- it never edits the base class.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# stable_baselines3/sb3_contrib bundle their own OpenMP runtime, which
# conflicts with the one numpy/scipy already loaded on macOS conda builds and
# aborts the process. This must be set before those libraries are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.ai_models.model_api import _labels_and_scores_from_predictions  # noqa: E402
from src.cognitive_models.cognitive_models.CoXAM.rl_agents.meta_policy_strategy import (  # noqa: E402
    CONDITIONS,
    CombinedBundle,
    CombinedPolicyConfig,
    CombinedStrategyPolicyEnv,
    load_sub_policies,
)
from src.cognitive_models.cognitive_models.CoXAM.utils import (  # noqa: E402
    AIDatasetLoader,
    DecisionTreeInterpreter,
    LogisticRegressionInterpreter,
    filter_by_app_and_model,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
COXAM_DATA_DIR = REPO_ROOT / "assets" / "ai_dataset" / "CoXAM"
COXAM_EXPLANATIONS_DIR = REPO_ROOT / "assets" / "explanations" / "CoXAM"
TRAINED_POLICIES_ROOT = (
    REPO_ROOT / "src" / "cognitive_models" / "cognitive_models" / "CoXAM" / "trained_policies"
)
UNIFIED_STRATEGY_ROOT = TRAINED_POLICIES_ROOT / "unified_strategy_policy"
META_POLICY_PATH = (
    TRAINED_POLICIES_ROOT / "combined_strategy_meta_policy" / "mixed" / "models" / "best_model.zip"
)

#: ``expMethod`` marker of the shared AI-prediction rows in a combined table.
PREDICTION_ONLY_METHOD = "__prediction_only__"

#: Feature order of the published CoXAM corpus, per dataset -- i.e. what
#: ``a0..a5`` mean in ``assets/explanations/CoXAM/*.csv``. Read out of
#: ``assets/ai_dataset/CoXAM/metadata.csv`` and pinned here because the corpus
#: is a fixed published artefact while the loaders are not.
#:
#: These are the two datasets CoXAM's own studies and parameter sweeps ran
#: against (``outputs/parameter_effect_sweeps/*.csv`` carries exactly
#: ``wine_quality`` and ``mushrooms``).
#:
#: **They do not match this repo's own loaders**, in membership or in order:
#:
#:   wine_quality  corpus: ..., Vinegar Taint, pH, Chlorides
#:                 loader: Alcohol, Vinegar Taint, Sulphates, SO2, pH  (no Chlorides)
#:   mushrooms     corpus: Bruises, Height, Width, Shape, Cap Diameter, Gill
#:                 loader: Bruises, Cap Diameter, Width, Height, Ring
#:
#: So a corpus explanation's ``a1`` is Sulphates while the study's own column 1
#: is Vinegar Taint. Feeding one to the other silently permutes the features,
#: which is why ``source="assets"`` is validated against this table rather than
#: trusted. ``source="fit"`` is unaffected: it derives everything from the
#: study's own dataset.
#: assets/ai_dataset/CoXAM/none.csv also carries predictions for
#: ``german_credit`` and ``diabetes``, but neither has any rows in
#: assets/explanations/CoXAM/{decision_tree,logistic_regression}.csv (0 vs.
#: ~2-3 for each dataset below) -- predictions with no surrogate to pair them
#: with, so ``source="assets"`` cannot serve those two; they still need
#: ``source="fit"``.
COXAM_CORPUS_FEATURES: dict[str, tuple[str, ...]] = {
    "wine_quality": ("Alcohol", "Sulphates", "SO2", "Vinegar Taint", "pH", "Chlorides"),
    "mushrooms": ("Bruises", "Height", "Width", "Shape", "Cap Diameter", "Gill"),
    "forest_cover": (
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
    ),
    "adult": ("Age", "Capital Gain", "Married", "Years of Education", "Sex", "Capital Loss"),
}

#: Features the dataset loader names differently from the published corpus.
#:
#: These are **confirmed equivalences, not guesses**: the corpus abbreviates the
#: loader's ``Gill Spacing`` to ``Gill``, capitalises ``chlorides``, and shortens
#: ``Marital Status`` to ``Married``. Anything not listed here is treated as a
#: genuinely different feature, because the corpus's ``a0..a5`` are positional
#: -- quietly accepting a near-match would attach coefficients and tree
#: thresholds to the wrong column.
COXAM_CORPUS_FEATURE_ALIASES: dict[str, dict[str, str]] = {
    "wine_quality": {"chlorides": "Chlorides"},
    "mushrooms": {"Gill Spacing": "Gill"},
    "adult": {"Marital Status": "Married"},
}


def coxam_loader_feature_cols(app_id: str) -> list[str]:
    """The corpus's feature set, spelled the way ``prepare_dataset`` needs.

    The inverse of :func:`corpus_feature_names`: given the corpus's own
    spelling and order (``COXAM_CORPUS_FEATURES``), returns what to pass as
    ``feature_cols`` so the prepared dataset matches the corpus exactly --
    which is what lets ``source="assets"`` run with no AI training. Order
    matters as much as spelling, since the corpus's ``a0..a5`` are positional;
    pass this together with ``rank_features_by_target=False``, or
    ``prepare_dataset`` will reorder it by target correlation.

    Raises:
        KeyError: If ``app_id`` is not in the published corpus.
    """
    corpus = COXAM_CORPUS_FEATURES[app_id]
    reverse = {corpus_name: loader_name for loader_name, corpus_name in
               COXAM_CORPUS_FEATURE_ALIASES.get(app_id, {}).items()}
    return [reverse.get(name, name) for name in corpus]


def corpus_feature_names(app_id: str, study_feature_names: Sequence[str]) -> tuple[str, ...]:
    """The study's feature names in the corpus's own spelling.

    Names with no alias are returned unchanged, so a real mismatch still shows
    up as a mismatch rather than being silently renamed.
    """
    aliases = COXAM_CORPUS_FEATURE_ALIASES.get(app_id, {})
    return tuple(aliases.get(str(name), str(name)) for name in study_feature_names)

#: How the design's xai_type vocabulary maps onto CoXAM's own CONDITIONS
#: vocabulary (support_matrix.json declares "decision_tree" |
#: "logistic_regression" | "hybrid"; CoXAM's CONDITIONS uses
#: "linear_regression" instead of "logistic_regression").
_CONDITION_ALIASES = {
    "decision_tree": "decision_tree",
    "logistic_regression": "linear_regression",
    "linear_regression": "linear_regression",
    "hybrid": "hybrid",
}

#: Per-trial explanation family CoXAM's episode schedule expects.
_EXPLANATION_TYPE_ALIASES = {
    "decision_tree": "dt",
    "logistic_regression": "lr",
    "logistic_regression_weights": "lr",
    "decision_list": "dt",
    "rule_list": "dt",
    "rule_set": "dt",
}

_FEATURE_COLUMN_RE = re.compile(r"^x(\d+)(_min|_max)?$")


def _canonical_coxam_condition(value: Any) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key not in _CONDITION_ALIASES:
        raise ValueError(
            f"Unknown CoXAM condition {value!r}. Use one of {sorted(_CONDITION_ALIASES)}."
        )
    return _CONDITION_ALIASES[key]


def _canonical_coxam_explanation_type(xai_method: Any, *, shown: bool) -> str:
    """Per-trial ``none`` | ``lr`` | ``dt``, matching the env's own vocabulary."""
    if not shown:
        return "none"
    key = str(xai_method).strip().lower().replace(" ", "_")
    if key not in _EXPLANATION_TYPE_ALIASES:
        raise ValueError(
            f"Unknown CoXAM xai_method {xai_method!r}. Use one of {sorted(_EXPLANATION_TYPE_ALIASES)}."
        )
    return _EXPLANATION_TYPE_ALIASES[key]


def default_coxam_config(**overrides: Any) -> CombinedPolicyConfig:
    """Base config pointed at the relocated trained sub-policies.

    Mirrors ``_default_config()`` in ``meta_policy_strategy_dashboard.py``.
    """
    base = dict(
        data_dir=str(COXAM_DATA_DIR),
        output_root=str(TRAINED_POLICIES_ROOT),
        run_name="coxam_study_runner",
        condition_name="mixed",
        instances_per_episode=1,
        lr_calculation_model_path=str(
            UNIFIED_STRATEGY_ROOT / "lr_calculation" / "models" / "final_model.zip"
        ),
        lr_heuristic_model_path=str(
            UNIFIED_STRATEGY_ROOT / "lr_heuristic" / "models" / "final_model.zip"
        ),
        dt_traversal_model_path=str(
            UNIFIED_STRATEGY_ROOT / "dt_traversal" / "models" / "final_model.zip"
        ),
    )
    base.update(overrides)
    return CombinedPolicyConfig(**base)


def load_coxam_meta_policy(path: Optional[Path] = None):
    """Load the trained meta-policy, ``MaskablePPO`` first with a ``PPO`` fallback.

    Mirrors ``ExperimentService.__init__`` in ``meta_policy_strategy_dashboard.py``.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO

    model_path = path or META_POLICY_PATH
    try:
        return MaskablePPO.load(model_path, device="cpu"), True
    except ValueError:
        return PPO.load(model_path, device="cpu"), False


def load_coxam_sub_policies(config: Optional[CombinedPolicyConfig] = None) -> dict[str, Any]:
    return load_sub_policies(config or default_coxam_config())


def _rename_dataset_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``dataId`` -> ``appId``, ``x{i}[_min|_max]`` -> ``v{i}[_min|_max]``.

    Only renames the numeric feature-bound columns ``LogisticRegressionInterpreter``
    reads (``v{i}_min``/``v{i}_max``); category-label columns (``x{i}_0``, ...) are
    display-only in the interpreter classes (used by ``_format_feature``, never by
    ``apply_to_instance``) and are left as-is since nothing here reads them.
    """
    rename = {"dataId": "appId"}
    for column in df.columns:
        match = _FEATURE_COLUMN_RE.match(column)
        if match:
            index, suffix = match.group(1), match.group(2) or ""
            rename[column] = f"v{index}{suffix}"
    return df.rename(columns=rename)


def coxam_metadata_table() -> pd.DataFrame:
    """Dataset-level per-feature bounds from the shared, vendored corpus.

    Not used by :func:`fit_coxam_surrogates` -- that function computes its own
    metadata from the study's own fitting sample. Kept available for callers
    that want the static, larger-sample bounds instead.
    """
    return _rename_dataset_level_columns(pd.read_csv(COXAM_DATA_DIR / "metadata.csv"))


def fit_coxam_surrogates(
    *,
    app_id: str,
    model_name: str,
    X: np.ndarray,
    instance_ids: Sequence[int],
    trained_ai_model: Any,
    feature_names: Sequence[str],
    ai_model_input: Optional[np.ndarray] = None,
) -> dict[str, pd.DataFrame]:
    """Fit fresh decision-tree and logistic-regression surrogates for a bundle.

    Mirrors ``tutorials/decision_tree_logistic_regression_experiment_workflow.ipynb``'s
    own pattern: predict with the study's trained AI model, then fit
    ``create_xai_method("rules"/"weights", ...)`` against those predictions.

    ``X`` must be **raw** feature values (``PreparedDataset.X_test_raw``, not
    ``X_test``): CoXAM's env only ever calls
    ``LogisticRegressionInterpreter``/``DecisionTreeInterpreter``'s
    ``apply_to_instance()`` with raw, un-normalized values (confirmed at
    ``meta_policy_strategy.py``'s ``lr_calculation``/``dt_traverse`` branches,
    which read ``self.X_raw[step_idx]``, and in ``_xai_match_pools``, which
    loads with ``normalize=False``). Fitting against normalized values would
    calibrate coefficients/thresholds to the wrong scale. Pass
    ``ai_model_input`` (the model-ready matrix, e.g. ``X_test``) separately
    when the trained AI model needs different-shaped input than the raw
    surrogate does (normalization, one-hot encoding); it defaults to ``X``
    when the AI model accepts raw input directly.

    This is deliberately independent of ``study.combined_explanations`` --
    that table (from the generic ``study.explanations(...)`` stage) carries
    the ``a{i}_i`` attribution schema shared with CoAX, produced by
    ``to_explanation_df()``. CoXAM's own ``LogisticRegressionInterpreter``/
    ``DecisionTreeInterpreter`` classes (unmodified, from ``coxam.utils``) need
    the different ``coef_a{i}``/``tree_structure`` schema produced by
    ``to_explanation_table()``/``to_metadata_table()`` on the same adapter
    objects, so those must be fit here rather than read off the study.

    Both variants CoXAM's episode-level "complexity" axis needs are fit and
    concatenated: logistic regression's dense and sparse, decision tree's
    depth 2 and depth 3 (matching ``load_combined_bundles``'s own bundle
    shape, which needs both of each present in one table).
    """
    from src.xai_adapter import create_xai_method

    # predict() returns scores, not labels: the torch MLP returns the network's
    # raw output and the Keras one returns probabilities. Casting those to int
    # truncates every probability below 1.0 to class 0, which would fit both
    # surrogates against an almost entirely zero label vector. Use the same
    # conversion classification_metrics does.
    predictions = trained_ai_model.predict(X if ai_model_input is None else ai_model_input)
    y_surrogate, _scores = _labels_and_scores_from_predictions(predictions)

    # Both surrogates need two classes to fit against. An undertrained model
    # that predicts one class everywhere otherwise fails deep inside the
    # surrogate generator with no indication of the real cause.
    observed = np.unique(y_surrogate)
    if len(observed) < 2:
        raise ValueError(
            f"The trained AI model predicts only class {observed.tolist()} across all "
            f"{len(y_surrogate)} instances, so decision-tree and logistic-regression "
            "surrogates cannot be fitted. Train the model longer or check its quality "
            "before running a CoXAM simulation."
        )

    lr_tables = []
    metadata_df: Optional[pd.DataFrame] = None
    for variant in ("dense", "sparse"):
        method = create_xai_method(
            "weights",
            app_id=app_id,
            model_name=model_name,
            variant=variant,
            feature_names=list(feature_names),
        ).fit(X, y_surrogate)
        lr_tables.append(method.to_explanation_table())
        if metadata_df is None:
            metadata_df = method.to_metadata_table()
    lr_explanations = pd.concat(lr_tables, ignore_index=True)

    dt_tables = []
    for depth in (2, 3):
        method = create_xai_method(
            "rules",
            app_id=app_id,
            model_name=model_name,
            depth=depth,
            feature_names=list(feature_names),
        ).fit(X, y_surrogate)
        dt_tables.append(method.to_explanation_table())
    dt_explanations = pd.concat(dt_tables, ignore_index=True)

    predictions_df = pd.DataFrame(
        {
            "dataId": app_id,
            "modelName": model_name,
            "instanceId": list(instance_ids),
            "pred": y_surrogate,
        }
    )

    assert metadata_df is not None
    return {
        "metadata": _rename_dataset_level_columns(metadata_df),
        "lr_explanations": _rename_dataset_level_columns(lr_explanations),
        "dt_explanations": _rename_dataset_level_columns(dt_explanations),
        "predictions": _rename_dataset_level_columns(predictions_df),
    }


def _check_corpus_features_match(app_id: str, study_feature_names: Sequence[str]) -> None:
    """Refuse to mix corpus explanations with a differently-shaped study.

    The corpus's ``a{i}`` indices are positional. If the study's own feature
    order differs, every coefficient and every tree threshold silently lands on
    the wrong feature -- a wrong answer rather than an error, which is why this
    is checked rather than assumed.
    """
    corpus = COXAM_CORPUS_FEATURES.get(app_id)
    if corpus is None:
        return

    # Compare in the corpus's spelling: the loader names a few features
    # differently (see COXAM_CORPUS_FEATURE_ALIASES) without them being
    # different features.
    study = corpus_feature_names(app_id, study_feature_names)
    if study == corpus:
        return

    if set(study) == set(corpus):
        detail = f"same features in a different order (corpus {corpus}, study {study})"
    else:
        missing = [name for name in corpus if name not in study]
        extra = [name for name in study if name not in corpus]
        detail = (
            f"different features -- corpus has {len(corpus)} {corpus}, study has "
            f"{len(study)} {study}"
            + (f"; missing from the study: {missing}" if missing else "")
            + (f"; not in the corpus: {extra}" if extra else "")
        )

    raise ValueError(
        f"The published CoXAM corpus for appId={app_id!r} does not match this study's "
        f"features: {detail}. The corpus's a0..a{len(corpus) - 1} are positional, so "
        "using it here would attach every coefficient and tree threshold to the wrong "
        "feature. Pass source='fit' to fit surrogates against this study's own dataset."
    )


def load_coxam_surrogates_from_assets(
    *,
    app_id: str,
    study_feature_names: Optional[Sequence[str]] = None,
    assets_data_dir: Path = COXAM_DATA_DIR,
    assets_explanations_dir: Path = COXAM_EXPLANATIONS_DIR,
) -> dict[str, pd.DataFrame]:
    """Load pre-generated surrogates for ``app_id`` from the shared asset corpus.

    Same return shape as :func:`fit_coxam_surrogates`, but reads the tables
    ``tutorials/decision_tree_logistic_regression_experiment_workflow.ipynb``
    already produced and saved under ``assets/ai_dataset/CoXAM`` and
    ``assets/explanations/CoXAM``, instead of fitting fresh ones. Faster, but
    only covers the datasets that tutorial has been run against so far
    (confirmed: adult, diabetes, forest_cover, german_credit, loan_approval,
    mushrooms, wine_quality, each fit against an ``mlp`` AI model) -- use
    :func:`fit_coxam_surrogates` for any other dataset or AI model.

    Already in CoXAM-native schema (``coef_a{i}``, ``tree_structure``), like
    :func:`fit_coxam_surrogates`'s output -- only the ``dataId``->``appId``
    rename is needed, same as every other table :func:`build_coxam_bundle`
    consumes.
    """
    metadata_df = pd.read_csv(assets_data_dir / "metadata.csv")
    if app_id not in metadata_df["dataId"].astype(str).values:
        raise ValueError(
            f"No CoXAM asset metadata for appId={app_id!r}. Available: "
            f"{sorted(metadata_df['dataId'].astype(str).unique())}. Use "
            "fit_coxam_surrogates(...) for a dataset outside this list."
        )
    if study_feature_names is not None:
        _check_corpus_features_match(app_id, study_feature_names)
    values_df = pd.read_csv(assets_data_dir / "values.csv")
    predictions_df = pd.read_csv(assets_data_dir / "none.csv")
    lr_explanations = pd.read_csv(assets_explanations_dir / "logistic_regression.csv")
    dt_explanations = pd.read_csv(assets_explanations_dir / "decision_tree.csv")

    return {
        "metadata": _rename_dataset_level_columns(metadata_df),
        "features": _rename_dataset_level_columns(values_df),
        "predictions": _rename_dataset_level_columns(predictions_df),
        "lr_explanations": _rename_dataset_level_columns(lr_explanations),
        "dt_explanations": _rename_dataset_level_columns(dt_explanations),
    }


def coxam_available_instance_ids(
    app_id: str,
    *,
    assets_data_dir: Path = COXAM_DATA_DIR,
) -> list[int]:
    """Instance ids the pre-generated asset corpus can serve for ``app_id``.

    Pass to ``study.generate_trials(allowed_instance_ids=...)`` so trials stay
    runnable against ``source="assets"``. Mirrors
    ``CoAXAssetRepository.available_instance_ids()``, which the same parameter
    was built for.
    """
    values = _rename_dataset_level_columns(pd.read_csv(assets_data_dir / "values.csv"))
    scoped = values[values["appId"].astype(str) == str(app_id)]
    return sorted(scoped["instanceId"].dropna().astype(int).unique().tolist())


#: Instances per predicted class in CoXAM's own published corpus.
#: ``assets/ai_dataset/CoXAM/none.csv`` carries 400 instances per dataset at
#: 200/200 for wine_quality and 209/191 for mushrooms -- balanced on purpose.
COXAM_CORPUS_INSTANCES_PER_CLASS = 200


def coxam_balanced_instance_ids(
    study: Any,
    *,
    per_class: int = COXAM_CORPUS_INSTANCES_PER_CLASS,
    data: Optional[Any] = None,
) -> list[int]:
    """A class-balanced instance pool, the way CoXAM's own corpus was built.

    CoXAM's published corpus is balanced by construction -- 400 instances per
    dataset, ~50/50 by the AI's predicted class. A raw dataset split usually is
    not: this repo's ``wine_quality`` runs about 9% class 1, so nine trials in
    ten ask the participant to flip a prediction the model makes confidently,
    and both the counterfactual DV and any forward accuracy measure end up
    dominated by the majority class rather than by the explanation.

    Pass the result to ``study.generate_trials(allowed_instance_ids=...)``::

        study.generate_trials(
            allowed_instance_ids=coxam_balanced_instance_ids(study), ...
        )

    Returns fewer than ``2 * per_class`` ids when a class is too small to fill,
    taking every available instance of the scarcer class.
    """
    dataset = data if data is not None else study.data
    if dataset is None:
        raise RuntimeError("No prepared dataset available. Call study.prepare_dataset(...) first.")

    trained_ai_model = getattr(study, "trained_ai_model", None)
    if trained_ai_model is None:
        raise RuntimeError("No trained AI model available. Call study.train_AI_model(...) first.")

    predictions = trained_ai_model.predict(dataset.split.X_model)
    labels, _scores = _labels_and_scores_from_predictions(predictions)
    labels = np.asarray(labels).reshape(-1)
    instance_ids = np.asarray(dataset.split.raw_instance_ids)

    selected: list[int] = []
    for label in np.unique(labels):
        matching = instance_ids[labels == label]
        selected.extend(int(value) for value in matching[:per_class])
    return sorted(selected)


def build_coxam_bundle(
    *,
    app_id: str,
    model_name: str,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    lr_explanations: pd.DataFrame,
    dt_explanations: pd.DataFrame,
    metadata: pd.DataFrame,
) -> CombinedBundle:
    """Build one ``CombinedBundle`` from already-CoXAM-native-schema tables.

    ``predictions``/``lr_explanations``/``dt_explanations``/``metadata`` are
    normally the output of :func:`fit_coxam_surrogates`. ``features`` is
    built separately from the study's own raw feature matrix (see
    ``coxam_study_runner.py``), since surrogate fitting has no need for it.
    Every table is renamed the same way (``dataId``->``appId``, ``x{i}``->
    ``v{i}``) so the unmodified ``coxam.utils`` classes can read them without
    any change to those classes; tables already in that shape are unaffected
    since the rename is a no-op on columns it doesn't recognize.
    """
    metadata_df = _rename_dataset_level_columns(metadata)
    features = _rename_dataset_level_columns(features)
    predictions = _rename_dataset_level_columns(predictions)
    lr_explanations = _rename_dataset_level_columns(lr_explanations)
    dt_explanations = _rename_dataset_level_columns(dt_explanations)

    loader = AIDatasetLoader(features, metadata_df, predictions)
    loader = filter_by_app_and_model(loader, app_id, model_name)
    lr_dense = LogisticRegressionInterpreter(lr_explanations, metadata_df, app_id, model_name, variant="dense")
    lr_sparse = LogisticRegressionInterpreter(lr_explanations, metadata_df, app_id, model_name, variant="sparse")
    dt_depth2 = DecisionTreeInterpreter(dt_explanations, metadata_df, app_id, model_name, depth=2)
    dt_depth3 = DecisionTreeInterpreter(dt_explanations, metadata_df, app_id, model_name, depth=3)

    feature_ids = set(loader.feature_values_df["instanceId"].dropna().astype(int).tolist())
    labeled_rows = loader.AI_predictions_df[loader.AI_predictions_df["pred"].notna()]
    prediction_ids = set(labeled_rows["instanceId"].dropna().astype(int).tolist())
    instance_ids = tuple(sorted(feature_ids & prediction_ids))
    if not instance_ids:
        raise ValueError(
            f"No instances with both features and a labelled AI prediction for "
            f"appId={app_id!r} modelName={model_name!r}."
        )
    return CombinedBundle(app_id, model_name, loader, lr_dense, lr_sparse, dt_depth2, dt_depth3, instance_ids)


@dataclass(frozen=True)
class ForcedEpisode:
    instance_ids: tuple[int, ...]
    explanation_schedule: tuple[str, ...]


class TrialDrivenCombinedStrategyPolicyEnv(CombinedStrategyPolicyEnv):
    """Forces a specific instance sequence instead of ``reset()`` sampling one.

    ``reset()`` (defined on the base class, unmodified) calls
    ``self._sample_instances(bundle)`` and ``self._make_explanation_schedule()``
    as overridable hook methods; this subclass overrides only those two so a
    study's trial order drives the episode, while every other mechanic
    (observation building, reward, history tracking, action masking) is
    inherited unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._forced: Optional[ForcedEpisode] = None

    def queue_forced_episode(self, *, instance_ids: list[int], explanation_schedule: list[str]) -> None:
        if len(instance_ids) != len(explanation_schedule):
            raise ValueError("instance_ids and explanation_schedule must be the same length.")
        self._forced = ForcedEpisode(tuple(instance_ids), tuple(explanation_schedule))

    def _sample_instances(self, bundle: CombinedBundle):
        if self._forced is None:
            return super()._sample_instances(bundle)
        chosen = list(self._forced.instance_ids)
        # Checked before load_instances, which would otherwise fail on an
        # unknown id with an opaque pandas IndexError.
        known = set(bundle.instance_ids)
        missing = [instance_id for instance_id in chosen if instance_id not in known]
        if missing:
            raise ValueError(
                f"Forced instance ids not in bundle {bundle.app_id}/{bundle.model_name}: {missing}"
            )
        raw_instances, labels = bundle.loader.load_instances(chosen, normalize=False)
        norm_instances, _ = bundle.loader.load_instances(chosen, normalize=True)
        pools = self._xai_match_pools(bundle)
        match_by_id = {
            instance_id: category
            for category, category_ids in pools.items()
            for instance_id in category_ids
        }
        self.sampled_instance_ids = chosen
        self.lr_explanation_matches_ai = np.asarray(
            [match_by_id[instance_id][0] for instance_id in chosen], dtype=bool
        )
        self.dt_explanation_matches_ai = np.asarray(
            [match_by_id[instance_id][1] for instance_id in chosen], dtype=bool
        )
        self.lr_explanation_episode_fidelity = float(self.lr_explanation_matches_ai.mean())
        self.dt_explanation_episode_fidelity = float(self.dt_explanation_matches_ai.mean())
        return (
            np.asarray(raw_instances, dtype=float),
            np.asarray(norm_instances, dtype=float),
            np.asarray(labels, dtype=int),
        )

    def _make_explanation_schedule(self) -> list[str]:
        if self._forced is None:
            return super()._make_explanation_schedule()
        return list(self._forced.explanation_schedule)


__all__ = [
    "COXAM_CORPUS_FEATURES",
    "COXAM_CORPUS_INSTANCES_PER_CLASS",
    "COXAM_DATA_DIR",
    "COXAM_EXPLANATIONS_DIR",
    "CONDITIONS",
    "PREDICTION_ONLY_METHOD",
    "TRAINED_POLICIES_ROOT",
    "ForcedEpisode",
    "TrialDrivenCombinedStrategyPolicyEnv",
    "build_coxam_bundle",
    "coxam_available_instance_ids",
    "coxam_balanced_instance_ids",
    "coxam_metadata_table",
    "fit_coxam_surrogates",
    "load_coxam_surrogates_from_assets",
    "default_coxam_config",
    "load_coxam_meta_policy",
    "load_coxam_sub_policies",
    "_canonical_coxam_condition",
    "_canonical_coxam_explanation_type",
]

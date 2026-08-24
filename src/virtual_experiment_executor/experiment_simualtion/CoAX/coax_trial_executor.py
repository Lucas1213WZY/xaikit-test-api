"""CoAX experiment execution helpers built on the current data-loader API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import warnings

import pandas as pd

from src.data_loaders import UnifiedDataLoader
from src.data_loaders.sources.coax_adapter import feature_prefix
from src.experiment_planner import select_trial_rows
from src.virtual_experiment_executor.participant_pools import provenance_columns


REPO_ROOT = Path(__file__).resolve().parents[4]
COAX_DATA_DIR = REPO_ROOT / "assets" / "ai_dataset" / "CoAX"
COAX_EXPLANATIONS_DIR = REPO_ROOT / "assets" / "explanations" / "CoAX"
COAX_MODEL_FILE = REPO_ROOT / "src" / "cognitive_models" / "cognitive_models" / "CoAX" / "coax_gcm_multiple_strategies.py"


def _coerce_bool(value: Any) -> bool:
    # tested_w_xai is a testing-phase IV, so training rows carry NaN. bool(nan) is
    # True in Python, which would silently treat every training row as tested with
    # XAI, so missing values resolve to False for the internal display decision.
    # The recorded column keeps training rows unset -- see run_coax_experiment_executor.
    if value is None or (isinstance(value, float) and value != value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "with", "w/ xai"}
    return bool(value)


def _canonical_xai_type(value: Any) -> str:
    key = str(value).strip().lower()
    aliases = {
        "none": "none",
        "no": "none",
        "no_xai": "none",
        "control": "none",
        "importance": "importance",
        "attribution": "attribution",
    }
    if key not in aliases:
        raise ValueError(f"Unknown XAI type {value!r}. Use none, importance, or attribution.")
    return aliases[key]


def _canonical_tested_condition(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _coerce_bool(value)


@lru_cache(maxsize=1)
def _load_coax_model_module():
    spec = importlib.util.spec_from_file_location("coax_gcm_multiple_strategies", COAX_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CoAX model module from {COAX_MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Every spelling a caller might use for a strategy name, resolved to the
#: class name as it appears in ``coax_gcm_multiple_strategies.py``.
COAX_STRATEGY_CLASS_NAMES: dict[str, str] = {
    "sensitivefeatures": "SensitiveFeatures",
    "sensitive_features": "SensitiveFeatures",
    "salientfeatures": "SalientFeatures",
    "salient_features": "SalientFeatures",
    "importancecategorization": "ImportanceCategorization",
    "importance_categorization": "ImportanceCategorization",
    "attributionsum": "AttributionSum",
    "attribution_sum": "AttributionSum",
}


def _normalize_strategy_class_name(strategy_name: str) -> str:
    """The class name for any accepted spelling of a strategy name."""
    normalized = str(strategy_name).strip().lower().replace("-", "_").replace(" ", "_")
    return COAX_STRATEGY_CLASS_NAMES.get(normalized, strategy_name)


#: Constructor parameters each strategy class actually reads, from its
#: ``__init__`` in ``coax_gcm_multiple_strategies.py`` -- excludes ``time``,
#: which the runner supplies itself rather than the UI/design layer.
#: ``AttributionSum`` is the only one with ``scaling_factor``, since it is the
#: only strategy whose response comes from a logistic over a summed
#: attribution rather than exemplar retrieval.
COAX_STRATEGY_PARAMS: dict[str, set[str]] = {
    "SensitiveFeatures": {"decay_param", "retrieval_threshold", "sensitivity", "k"},
    "SalientFeatures": {"decay_param", "retrieval_threshold", "sensitivity", "k"},
    "ImportanceCategorization": {"decay_param", "retrieval_threshold", "sensitivity", "k"},
    "AttributionSum": {
        "decay_param", "retrieval_threshold", "sensitivity", "scaling_factor", "k", "explanation_type",
    },
}


def coax_params_for_strategy(strategy_name: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the ``overrides`` entries ``strategy_name``'s constructor reads.

    A design's cognitive-parameter config is one flat dict applied to every
    condition (see ``coax_models_for_trials``), and conditions run different
    strategy classes -- ``SensitiveFeatures`` has no ``scaling_factor``,
    ``AttributionSum``'s ``k`` selects top-k attributions rather than top-k
    retrieved exemplars, and a mistyped or wrong-agent label (e.g. a
    CoXAM/Sim2Real parameter name that slipped into a CoAX design) should not
    silently reach a strategy that has no use for it. Every strategy class
    accepts ``**kwargs`` so an unfiltered override would not raise -- it would
    just be ignored -- but filtering here makes the routing explicit rather
    than relying on that.

    An unknown ``strategy_name`` passes every override through unfiltered,
    since :func:`make_coax_model` is what raises on an unknown strategy.
    """
    accepted = COAX_STRATEGY_PARAMS.get(_normalize_strategy_class_name(strategy_name))
    if accepted is None:
        return dict(overrides)
    return {key: value for key, value in overrides.items() if key in accepted}


def make_coax_model(strategy_name: str, **strategy_kwargs):
    """Instantiate one of the CoAX strategy classes from the model file."""
    module = _load_coax_model_module()
    class_name = _normalize_strategy_class_name(strategy_name)
    try:
        strategy_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown CoAX strategy {strategy_name!r}.") from exc
    return strategy_cls(**strategy_kwargs)


# Which strategies the CoAX study allows per condition. Mirrors STRATS_BY_XAI and
# available_strategies() in generate_strategy_sample_from_params.py, the module the
# mockup server drives, so notebook runs and the simulator agree on what is legal.
COAX_STRATEGIES_BY_XAI_TYPE: dict[str, list[str]] = {
    "none": ["SensitiveFeatures"],
    "importance": [
        "SensitiveFeatures",
        "SalientFeatures",
        "ImportanceCategorization",
        "AttributionSum",
    ],
    "attribution": ["SensitiveFeatures", "AttributionSum"],
}

# Strategies the reference pipeline removes for a specific tested condition.
COAX_STRATEGY_EXCLUSIONS: dict[tuple[str, bool], set[str]] = {
    ("importance", False): {"ImportanceCategorization"},
    ("attribution", True): {"SensitiveFeatures"},
}

# Slider limits the simulator clamps every request to (PARAM_BOUNDS in the
# reference module). Fitted per-participant values in results/ range wider than
# this, so these are UI bounds, not the bounds of the fitted population.
COAX_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "k": (1, 5),
    "sensitivity": (1.0, 20.0),
    "retrieval_threshold": (-2.3, -1.3),
    "scaling_factor": (1.0, 7.0),
}

# Defaults taken from the reference CLI (sensitivity, scaling_factor, decay_param)
# and from the simulator's demo cases for the two arguments the CLI requires
# (k, retrieval_threshold). AttributionSum takes scaling_factor and no sensitivity;
# every other strategy takes sensitivity and no scaling_factor, as in strategy_config().
DEFAULT_COAX_PARAMS: dict[str, Any] = {
    "decay_param": 0.5,
    "k": 3,
    "retrieval_threshold": -2.3,
}
DEFAULT_COAX_SENSITIVITY = 10.0
DEFAULT_COAX_SCALING_FACTOR = 1.0


# The strategy each explanation type is meant to exercise. Unlike the reference
# module -- which just takes the first entry of COAX_STRATEGIES_BY_XAI_TYPE, and so
# lands on SensitiveFeatures almost everywhere -- this keeps one distinct strategy
# per explanation type. It falls back to a legal strategy where the preferred one
# is excluded for that tested condition.
PREFERRED_COAX_STRATEGY_BY_XAI_TYPE: dict[str, str] = {
    "none": "SensitiveFeatures",
    "importance": "SalientFeatures",
    "attribution": "AttributionSum",
}


def coax_available_strategies(xai_type: Any, tested_w_xai: Any = None) -> list[str]:
    """Strategies the CoAX study allows for a condition."""
    key = _canonical_xai_type(xai_type)
    strategies = list(COAX_STRATEGIES_BY_XAI_TYPE[key])
    if tested_w_xai is None:
        return strategies
    excluded = COAX_STRATEGY_EXCLUSIONS.get((key, _canonical_tested_condition(tested_w_xai)), set())
    return [name for name in strategies if name not in excluded]


def default_coax_strategy(xai_type: Any, tested_w_xai: Any = None) -> str:
    """The preferred strategy for a condition, or the first legal one if excluded."""
    allowed = coax_available_strategies(xai_type, tested_w_xai)
    preferred = PREFERRED_COAX_STRATEGY_BY_XAI_TYPE[_canonical_xai_type(xai_type)]
    return preferred if preferred in allowed else allowed[0]


def default_coax_params(strategy_name: str, xai_type: Any, **overrides: Any) -> dict[str, Any]:
    """Reference-shaped constructor kwargs for one strategy under one condition."""
    params = dict(DEFAULT_COAX_PARAMS)
    if strategy_name == "AttributionSum":
        params["scaling_factor"] = DEFAULT_COAX_SCALING_FACTOR
        params["explanation_type"] = _canonical_xai_type(xai_type)
    else:
        params["sensitivity"] = DEFAULT_COAX_SENSITIVITY
    params.update(overrides)
    return params


def make_coax_models(
    strategies: Optional[Mapping[Any, str]] = None,
    params: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    *,
    by_tested_condition: bool = False,
) -> dict[Any, Any]:
    """Build one CoAX strategy per condition, keyed the way the trial table names it.

    The returned mapping is what ``run_coax_experiment_executor`` expects when it
    should pick the model itself instead of being handed a single strategy. Keys are
    ``xai_type`` strings, or ``(xai_type, tested_w_xai)`` tuples when
    ``by_tested_condition`` is set — the reference pipeline varies the legal strategy
    set by tested condition, so tuple keys are what reproduce it exactly.
    """
    overrides = {_normalize_model_key(key): name for key, name in (strategies or {}).items()}
    param_overrides = {_normalize_model_key(key): dict(value) for key, value in (params or {}).items()}

    keys: list[Any]
    if by_tested_condition:
        keys = [(xai_type, tested) for xai_type in COAX_STRATEGIES_BY_XAI_TYPE for tested in (False, True)]
    else:
        keys = list(COAX_STRATEGIES_BY_XAI_TYPE)

    models: dict[Any, Any] = {}
    for key in keys:
        xai_type, tested = key if isinstance(key, tuple) else (key, None)
        strategy_name = overrides.get(key) or default_coax_strategy(xai_type, tested)
        filtered = coax_params_for_strategy(strategy_name, param_overrides.get(key, {}))
        models[key] = make_coax_model(
            strategy_name,
            **default_coax_params(strategy_name, xai_type, **filtered),
        )
    return models


_SHARED_MODEL_KEY = "__shared__"


def _normalize_model_key(key: Any) -> Any:
    """Canonicalize an ``xai_type`` or ``(xai_type, tested_w_xai)`` mapping key."""
    if isinstance(key, tuple):
        if len(key) != 2:
            raise ValueError(f"Model key {key!r} must be (xai_type, tested_w_xai).")
        return (_canonical_xai_type(key[0]), _canonical_tested_condition(key[1]))
    return _canonical_xai_type(key)


def _select_model_key(model_templates: Mapping[Any, Any], xai_type: str, tested_w_xai: bool) -> Any:
    """Pick the registered model for a trial: exact condition first, then xai_type."""
    if _SHARED_MODEL_KEY in model_templates:
        return _SHARED_MODEL_KEY
    for key in ((xai_type, tested_w_xai), xai_type):
        if key in model_templates:
            return key
    raise ValueError(
        f"No CoAX model supplied for xai_type={xai_type!r} tested_w_xai={tested_w_xai!r}. "
        f"Registered keys: {sorted(model_templates, key=repr)}."
    )


def _resolve_model_templates(cognitive_model: Any) -> dict[Any, Any]:
    """Normalize the ``cognitive_model`` argument into a condition -> model map.

    A single strategy is stored under a shared key so it keeps one memory across
    every condition, exactly as it behaved before per-condition dispatch existed.
    """
    if not isinstance(cognitive_model, Mapping):
        return {_SHARED_MODEL_KEY: cognitive_model}
    if not cognitive_model:
        raise ValueError("cognitive_model mapping is empty; pass at least one model per condition.")
    return {_normalize_model_key(key): model for key, model in cognitive_model.items()}


def _explanation_value_columns(df: pd.DataFrame) -> list[str]:
    """Attribution columns of a CoAX-schema explanation table (a0_i, a1_i, ...)."""
    return [c for c in df.columns if c.startswith("a") and c.endswith("_i")]


#: The published CoAX corpus's own feature set per dataset, in the order its
#: x0..x4 columns are positional against (assets/ai_dataset/CoAX/metadata.csv)
#: -- confirmed to already match the data loader's own spelling exactly for
#: all three datasets (unlike CoXAM's corpus, which needs an alias table). A
#: freshly trained model must use exactly this feature set for its
#: predictions/explanations to mean the same thing the corpus's positional
#: a0..a4 attributions do; a generic top-N-by-correlation default can select
#: entirely different columns (confirmed for forest_cover: only 2 of 5
#: overlapped), producing model predictions unrelated to what CoAX represents.
COAX_CORPUS_FEATURES: dict[str, tuple[str, ...]] = {
    "forest_cover": (
        "Elevation", "Aspect", "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am",
    ),
    "adult": ("Age", "Years of Education", "Marital Status", "Sex", "Capital Gain"),
    "wine_quality": ("Vinegar Taint", "SO2", "pH", "Sulphates", "Alcohol"),
}


def coax_loader_feature_cols(data_id: str) -> list[str]:
    """The corpus's feature set for ``data_id``, in its positional order.

    Pass as ``feature_cols`` (with ``rank_features_by_target=False``) to
    ``prepare_dataset`` so a freshly trained model's feature space matches the
    corpus exactly. Mirrors ``coxam_loader_feature_cols``.

    Raises:
        KeyError: If ``data_id`` is not in the published corpus.
    """
    return list(COAX_CORPUS_FEATURES[data_id])


#: Which XAI method the published CoAX corpus's explanation tables were built
#: with, per dataset -- confirmed against attribution.csv/importance.csv
#: themselves: both tables use the *same* method for a given dataset (adult
#: and wine_quality: lime; forest_cover: shap), so the method is a property of
#: the dataset/model, not of the xai_type (none/importance/attribution) shown.
#: A design that names xai_type but not xai_method is otherwise ambiguous --
#: run_coax_study cannot tell, per trial, which of multiple generated methods
#: to display -- so this resolves it the same way the corpus itself did.
COAX_CORPUS_XAI_METHOD: dict[str, str] = {
    "adult": "lime",
    "forest_cover": "shap",
    "wine_quality": "lime",
}


class CoAXAssetRepository:
    """Serve CoAX feature values, AI predictions, and explanations to the executor.

    Two sources are supported, because the same schema can come from either:

    * ``from_assets`` (the default) reads the fixed CoAX study files under
      ``assets/ai_dataset/CoAX`` and ``assets/explanations/CoAX``.
    * ``from_tables`` takes DataFrames, so tables produced by ``xai_adapter`` for
      any dataset can drive the same executor. Those tables already share the
      CoAX schema, ``pred`` column included, so no conversion is needed.
    """

    def __init__(
        self,
        data_dir: Path = COAX_DATA_DIR,
        explanations_dir: Path = COAX_EXPLANATIONS_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.explanations_dir = Path(explanations_dir)
        self._features: pd.DataFrame = pd.DataFrame()
        self._predictions: pd.DataFrame = pd.DataFrame()
        self._explanations: dict[str, pd.DataFrame] = {}
        self._explanation_columns: dict[str, list[str]] = {}
        self._load_from_assets()

    @classmethod
    def from_assets(
        cls,
        data_dir: Path = COAX_DATA_DIR,
        explanations_dir: Path = COAX_EXPLANATIONS_DIR,
    ) -> "CoAXAssetRepository":
        """Build from the fixed CoAX study files on disk."""
        return cls(data_dir=data_dir, explanations_dir=explanations_dir)

    @classmethod
    def from_tables(
        cls,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        explanations: Optional[Mapping[Any, pd.DataFrame]] = None,
    ) -> "CoAXAssetRepository":
        """Build from in-memory tables, e.g. those generated by ``xai_adapter``.

        ``features`` needs ``instanceId`` plus feature columns (``x0..`` or
        ``v0..``); ``predictions`` needs ``instanceId`` and ``pred``;
        ``explanations`` maps an xai_type to a table carrying ``a0_i``-style
        columns. A ``dataId`` column is honoured on any of them when present.
        """
        repo = cls.__new__(cls)
        repo.data_dir = None
        repo.explanations_dir = None
        repo._features = features.copy()
        repo._predictions = predictions.copy()
        repo._explanations = {}
        repo._explanation_columns = {}
        for xai_type, table in (explanations or {}).items():
            key = _canonical_xai_type(xai_type)
            if key == "none":
                continue
            repo._explanations[key] = table.copy()
            repo._explanation_columns[key] = _explanation_value_columns(table)
        return repo

    def _load_from_assets(self) -> None:
        feature_file = self.data_dir / "values.csv"
        metadata_file = self.data_dir / "metadata.csv"
        table_names = {
            "none": "none.csv",
            "importance": "importance.csv",
            "attribution": "attribution.csv",
        }

        for xai_type, file_name in table_names.items():
            explanation_file = self.explanations_dir / file_name
            explanation_df = pd.read_csv(explanation_file)
            explanation_columns = (
                [] if xai_type == "none" else _explanation_value_columns(explanation_df)
            )
            loader = UnifiedDataLoader.from_coax(
                feature_file=str(feature_file),
                metadata_file=str(metadata_file),
                prediction_file=str(explanation_file),
                explanation_columns=explanation_columns,
            )
            if xai_type == "none":
                # none.csv is the AI-prediction tracking record for the study.
                self._features = loader.get_feature_values()
                self._predictions = loader.get_ai_predictions()
            else:
                self._explanations[xai_type] = explanation_df
                self._explanation_columns[xai_type] = explanation_columns

    def available_instance_ids(self, data_id: Optional[str] = None) -> list[int]:
        """Instance IDs this repository can serve, for constraining trial generation."""
        df = self._filter_data_id(self._features, data_id)
        return sorted(df["instanceId"].astype(int).unique().tolist())

    @staticmethod
    def _filter_data_id(df: pd.DataFrame, data_id: Optional[str]) -> pd.DataFrame:
        """Scope a table to one dataset, under either spelling of the column.

        ``appId`` and ``dataId`` are the same thing -- the dataset's name. The
        CoAX study files spell it ``appId`` and the tables ``xai_adapter``
        produces spell it ``dataId``, so a filter that knew only one of them
        silently matched nothing when handed the other: every dataset in a
        multi-dataset corpus then collapsed onto whichever row for that
        instance id came first.
        """
        if data_id is None:
            return df
        for column in ("dataId", "appId"):
            if column in df.columns:
                return df[df[column].astype(str) == str(data_id)]
        return df

    def _row_for(self, df: pd.DataFrame, instance_id: int, data_id: Optional[str], what: str):
        scoped = self._filter_data_id(df, data_id)
        match = scoped[scoped["instanceId"].astype(int) == int(instance_id)]
        if match.empty:
            raise ValueError(
                f"Instance {instance_id} not found for dataId={data_id!r} in CoAX {what}."
            )
        return match.iloc[0]

    def get_trial_payload(
        self,
        instance_id: int,
        xai_type: Any,
        data_id: Optional[str] = None,
    ) -> tuple[list[float], Any, Optional[list[float]]]:
        loader_key = _canonical_xai_type(xai_type)

        feature_row = self._row_for(self._features, instance_id, data_id, "features")
        prediction_row = self._row_for(self._predictions, instance_id, data_id, "predictions")

        prefix = feature_prefix(self._features)
        features = []
        i = 0
        while True:
            column = f"{prefix}{i}"
            if column not in feature_row.index or pd.isna(feature_row[column]):
                break
            features.append(float(feature_row[column]))
            i += 1

        # The 'none' condition has no explanation table, so the lookup is skipped
        # rather than attempted and discarded.
        explanation: Optional[list[float]] = None
        if loader_key != "none":
            if loader_key not in self._explanations:
                raise ValueError(
                    f"No explanation table for xai_type={loader_key!r}. "
                    f"Available: {sorted(self._explanations)}."
                )
            table = self._filter_data_id(self._explanations[loader_key], data_id)
            match = table[table["instanceId"].astype(int) == int(instance_id)]
            row = {} if match.empty else match.iloc[0].to_dict()
            explanation = [row.get(column) for column in self._explanation_columns[loader_key]]

        return features, prediction_row.get("pred"), explanation


def coax_available_instance_ids(
    data_id: Optional[str] = None,
    *,
    data_dir: Path = COAX_DATA_DIR,
    explanations_dir: Path = COAX_EXPLANATIONS_DIR,
) -> list[int]:
    """Instance ids the published CoAX corpus can serve, optionally for one dataset.

    Pass to ``study.generate_trials(allowed_instance_ids=...)`` so trials stay
    runnable against ``source="corpus"``/``source="study"`` reading the
    corpus's own predictions. A thin module-level wrapper around
    ``CoAXAssetRepository.available_instance_ids()`` -- that class needs a
    live repository instance already loaded from assets; this is the
    zero-setup entry point ``coxam_available_instance_ids`` already has.
    """
    return CoAXAssetRepository(data_dir=data_dir, explanations_dir=explanations_dir).available_instance_ids(data_id)


@dataclass
class SimulationClock:
    """Simple time helper matching the CoAX model contract."""

    current_time: float = 0.0

    def get_time(self) -> float:
        return float(self.current_time)

    def add_time(self, amount: float) -> None:
        self.current_time += float(amount)


def _argmax_choice(probs: Any) -> Any:
    if not probs:
        return None
    return max(sorted(probs.keys()), key=lambda label: probs[label])


def run_coax_experiment_executor(
    trials: list[dict[str, Any]] | pd.DataFrame,
    cognitive_model: Any,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    data_repository: Optional[CoAXAssetRepository] = None,
    trial_order_columns: Optional[list[str]] = None,
    output_probabilities: bool = True,
    train_with_explanation: bool = True,
    dvs: Optional[Mapping[str, Any]] = None,
    participant_models: Optional[Callable[[Any], Any]] = None,
    participant_draws: Optional[Mapping[Any, Any]] = None,
) -> pd.DataFrame:
    """Execute CoAX strategies over generated trial rows.

    ``cognitive_model`` is either a single strategy used for every trial, or a
    mapping of ``xai_type`` -> strategy (see :func:`make_coax_models`). With a
    mapping, each trial is routed to the model its own ``xai_type`` names, so the
    whole experiment runs in one call instead of one call per condition.

    Each trial yields one row per inference step, matching the CoAX study
    procedure: a training trial showing an explanation produces an
    ``infer_no_explanation`` row and an ``infer_with_explanation`` row, with the
    feedback time recorded on the second. ``train_with_explanation`` mirrors the
    flag of the same name in the reference runner.

    Pass ``dvs`` (e.g. ``study.DVs``) to fill accuracy DV columns using the same
    convention as ``run_experiment_executor``, so the result frame can be handed
    back to the study object for analysis and plotting.

    ``participant_models`` builds one participant's condition -> model mapping on
    demand, called once per participant, so every participant can run its own
    fitted parameters instead of the study's one shared set -- what
    ``mode="diverse_participant"`` uses. It wins over ``cognitive_model``.
    ``participant_draws`` carries the matching provenance, keyed by
    ``(participantId, model_key)``, and is stamped onto every result row.

    The runner stays separate from xaikitTest so the study object can remain
    focused on experimental design and trial generation.
    """
    model_templates = _resolve_model_templates(cognitive_model)

    if dvs is not None:
        scorable = [name for name in dvs if "accuracy" in name.lower()]
        if not scorable:
            warnings.warn(
                f"None of the DVs {sorted(dvs)} contain 'accuracy', so no DV column "
                "will be filled. Only `cognitive_correct_vs_ai` is recorded, and "
                "study.analyze_iv_dv()/plot_results_grid() will not find a DV to "
                "read.",
                stacklevel=2,
            )
        # Every scorable DV is filled with agent-vs-AI agreement below, which is
        # forward simulation. A DV named for a different task therefore gets
        # forward numbers under its own label -- correct-looking and wrong. A
        # design export coerces this away before reaching here; setting DVs
        # directly (a notebook) does not, so say so rather than let it pass.
        mislabelled = [name for name in scorable if name != "forward_accuracy"]
        if mislabelled:
            warnings.warn(
                f"CoAX simulates forward simulation only, but DV(s) {sorted(mislabelled)} "
                "will be filled with agent-vs-AI agreement -- i.e. forward-simulation "
                "values carrying another task's name. Use `forward_accuracy`, or read "
                "`cognitive_correct_vs_ai` directly.",
                stacklevel=2,
            )

    trials_df = pd.DataFrame(trials).copy()
    selected = select_trial_rows(
        trials_df,
        mode,
        participant_id=participant_id,
        condition_filter=condition_filter,
    ).copy()

    if selected.empty:
        return selected

    if data_repository is None:
        data_repository = CoAXAssetRepository()

    order_columns = trial_order_columns or [
        column for column in ["participantId", "trialId", "trialWithinBlock", "block"] if column in selected.columns
    ]
    if order_columns:
        selected = selected.sort_values(order_columns, kind="stable")

    results: list[dict[str, Any]] = []
    participant_groups = (
        selected.groupby("participantId", sort=False, dropna=False)
        if "participantId" in selected.columns
        else [(None, selected)]
    )

    for _participant_key, participant_rows in participant_groups:
        clock = SimulationClock()
        models_by_key: dict[Any, Any] = {}
        # Resolved per participant rather than once for the study: in diverse
        # mode each participant carries its own fitted parameters, so it needs
        # its own template set to deepcopy from.
        templates = (
            model_templates
            if participant_models is None
            else _resolve_model_templates(participant_models(_participant_key))
        )

        for _, row in participant_rows.iterrows():
            trial = row.to_dict()
            instance_id = int(trial["instanceId"])
            data_id = trial.get("dataId", trial.get("dataset", trial.get("dataset_id")))
            xai_type = _canonical_xai_type(trial.get("xai_type", trial.get("XAIType", "none")))
            phase = str(trial.get("phase", "testing")).lower().strip()
            tested_w_xai = _canonical_tested_condition(trial.get("tested_w_xai", trial.get("Tested w/ XAI", False)))
            with_explanation = phase == "training" or tested_w_xai

            model_key = _select_model_key(templates, xai_type, tested_w_xai)
            if model_key not in models_by_key:
                # One instance per participant so memory accumulates across their trials.
                model = deepcopy(templates[model_key])
                model.time = clock
                models_by_key[model_key] = model
            model = models_by_key[model_key]
            draw = (participant_draws or {}).get((_participant_key, model_key))

            model.new_instance()

            feature_values, ai_prediction, explanation = data_repository.get_trial_payload(instance_id, xai_type, data_id=data_id)

            # Assembled from the three sources the API already holds: dataset
            # features, the explanation table, and the generated trial row.
            def make_context(shown_explanation: Optional[list[float]]) -> dict[str, Any]:
                return {
                    "features": feature_values,
                    "explanation": shown_explanation,
                    "ai_prediction": ai_prediction,
                }

            def result_row(step: str, shown_explanation, response, infer_time) -> dict[str, Any]:
                probs = dict(response) if isinstance(response, dict) else response
                agent_prediction = _argmax_choice(probs) if isinstance(probs, dict) else probs
                correct_vs_ai = (
                    None
                    if ai_prediction is None or agent_prediction is None
                    else bool(int(agent_prediction) == int(ai_prediction))
                )
                # Same DV convention as run_experiment_executor, so these results
                # can be analysed and plotted through the study object.
                dv_columns = {
                    name: int(correct_vs_ai)
                    for name in (dvs or {})
                    if "accuracy" in name.lower() and correct_vs_ai is not None
                }
                return {
                    **dv_columns,
                    **trial,
                    **provenance_columns(draw),
                    "phase": phase,
                    "step": step,
                    # tested_w_xai is a testing-phase IV. Training rows leave it
                    # unset rather than reporting False, which would assert the
                    # participant was tested without XAI during training.
                    "tested_w_xai": None if phase == "training" else tested_w_xai,
                    "xai_type": xai_type,
                    "cognitive_model_strategy": type(model).__name__,
                    "feature_values": feature_values,
                    "explanation": shown_explanation,
                    "response": probs if output_probabilities else None,
                    "ai_prediction": ai_prediction,
                    "agent_prediction": agent_prediction,
                    "pred_time": infer_time,
                    "cognitive_correct_vs_ai": correct_vs_ai,
                }

            # Mirrors StrategyComparisonRunner.generalized_run_experiment: a training
            # trial that shows an explanation is answered twice -- once from the
            # instance alone, then again after the explanation is revealed -- and
            # feedback lands on both attempts.
            is_training = phase == "training"
            show_explanation = explanation is not None and (
                tested_w_xai if not is_training else train_with_explanation
            )

            if (not show_explanation) or is_training:
                response, infer_time = model.infer(make_context(None))
                results.append(result_row("infer_no_explanation", None, response, infer_time))

            if show_explanation:
                response, infer_time = model.infer(make_context(explanation))
                results.append(result_row("infer_with_explanation", explanation, response, infer_time))

            if is_training:
                feedback_context = make_context(explanation if show_explanation else None)
                results[-1]["feedback_time"] = model.feedback(feedback_context)

    unscored = sum(1 for row in results if row["cognitive_correct_vs_ai"] is None)
    if unscored:
        warnings.warn(
            f"{unscored} of {len(results)} recorded steps could not be scored "
            "against the AI because the trial had no AI prediction or the model "
            "returned none. Those rows carry no DV value.",
            stacklevel=2,
        )

    return pd.DataFrame(results)


__all__ = [
    "COAX_DATA_DIR",
    "COAX_EXPLANATIONS_DIR",
    "COAX_PARAM_BOUNDS",
    "COAX_STRATEGIES_BY_XAI_TYPE",
    "COAX_STRATEGY_CLASS_NAMES",
    "COAX_STRATEGY_EXCLUSIONS",
    "COAX_STRATEGY_PARAMS",
    "COAX_CORPUS_FEATURES",
    "COAX_CORPUS_XAI_METHOD",
    "DEFAULT_COAX_PARAMS",
    "PREFERRED_COAX_STRATEGY_BY_XAI_TYPE",
    "CoAXAssetRepository",
    "SimulationClock",
    "coax_available_instance_ids",
    "coax_available_strategies",
    "coax_loader_feature_cols",
    "coax_params_for_strategy",
    "default_coax_params",
    "default_coax_strategy",
    "make_coax_model",
    "make_coax_models",
    "run_coax_experiment_executor",
]
"""Draw per-participant cognitive parameters from the fitted human populations.

Every simulation mode before ``diverse_participant`` ran one parameter set for
the whole study, so all N virtual participants inside a condition were the same
person taking N sessions. The consequences are visible in the tutorial
notebooks: Sim2Real reports a within-condition SD of exactly 0.0 in three of
four conditions (its t statistics reach 1e15 and p underflows to zero), and
CoXAM reports p = 2.7e-10 for an effect whose only variance source is which
instances each participant happened to see.

This module supplies the missing individual differences from the place they
actually exist -- the parameters fitted to the real study participants, shipped
anonymised under ``assets/human_data/`` by ``assets/build_human_data.py``. Each
virtual participant is dealt one real participant's fitted row, filtered to the
condition cell the virtual participant occupies, so a "decision_tree / high
complexity" simulated participant gets parameters fitted to a human who was
actually in that cell.

``coax_study_runner.FITTED_COAX_PARAMS`` already predicted this fix in prose:
the fitted distributions are bimodal (41% of SensitiveFeatures participants are
pinned at the sensitivity floor of 1.0, the rest spread 19-100), so their means
describe nobody, and sampling the distribution is both more realistic *and*
better calibrated than any single shared value.

The pools are always read from ``assets/human_data`` rather than from the
originals under ``src/cognitive_models/``: that copy is renumbered and stripped
of raw participant ids (see ``assets/build_human_data.py``), which is the copy
this repository can publish.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Same root ``src/result_visualizer/study_comparisons.py`` reads.
HUMAN_DATA = REPO_ROOT / "assets" / "human_data"


# -- value canonicalization ----------------------------------------------
#
# Trial tables and fitted tables spell the same condition differently: a trial
# says ``dataId="wine_quality"`` where the counterfactual replay says
# ``"Wine Quality"``, and ``tested_w_xai=True`` where CoAX's fit says
# ``"w/ XAI"``. Both sides go through the same canonicalizer before matching,
# so a filter compares meanings rather than spellings.


def _text(value: Any) -> str:
    """Lowercase, underscore-joined form of any label."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def canon_dataset(value: Any) -> str:
    """``"Wine Quality"``, ``"wine_quality"`` and ``"WineQuality"`` all match."""
    text = _text(value)
    return {"winequality": "wine_quality", "forestcover": "forest_cover"}.get(text, text)


def canon_tested(value: Any) -> str:
    """``True`` / ``"w/ XAI"`` -> ``"true"``; ``False`` / ``"w/o XAI"`` -> ``"false"``.

    Order matters: ``w/o XAI`` canonicalizes to ``w_o_xai``, which *contains*
    ``w_`` -- so the negative is tested first.
    """
    text = _text(value)
    if text in {"false", "0", "no", "n", "w_o_xai", "wo_xai", "without_xai"}:
        return "false"
    if text in {"true", "1", "yes", "y", "w__xai", "w_xai", "with_xai"}:
        return "true"
    return text


def canon_coax_xai_type(value: Any) -> str:
    """CoAX's ``XAIType``: ``Attribution`` / ``Importance`` / blank (= none).

    The anonymised export writes the original ``"None"`` as an empty cell, so an
    empty value means the no-explanation condition rather than a missing value.
    """
    text = _text(value)
    if text in {"", "none", "nan", "no_xai", "control"}:
        return "none"
    return text


def canon_coxam_condition(value: Any) -> str:
    """CoXAM's condition axis, however the table spells it."""
    text = _text(value)
    return {
        "dt": "decision_tree",
        "lr": "linear_regression",
        "dt+lr": "hybrid",
        "dt_lr": "hybrid",
        "logistic_regression": "linear_regression",
        "weights": "linear_regression",
        "rules": "decision_tree",
    }.get(text, text)


#: CoAX's fitted ``Strategy`` labels -> the strategy class the runner builds.
_COAX_STRATEGY_LABELS: dict[str, str] = {
    "attribution_sum": "AttributionSum",
    "sensitive_features_categorization": "SensitiveFeatures",
    "salient_features_categorization": "SalientFeatures",
    "importance_categorization": "ImportanceCategorization",
}


def canon_coax_strategy(value: Any) -> str:
    """``"Sensitive-features categorization"`` -> ``"SensitiveFeatures"``."""
    text = _text(value)
    return _COAX_STRATEGY_LABELS.get(text, str(value).strip())


def canon_plain(value: Any) -> str:
    """Lowercased text, for axes that need no aliasing (complexity, property)."""
    return _text(value)


# -- pool specifications --------------------------------------------------


@dataclass(frozen=True)
class PoolFilter:
    """One condition axis: what the runner calls it, what the CSV calls it."""

    key: str
    column: str
    canon: Callable[[Any], str] = canon_plain


@dataclass(frozen=True)
class PoolSpec:
    """Where one framework's fitted parameters live and how to read them.

    ``parameters`` maps CSV column -> the keyword the runner passes to its own
    model or environment. ``ranges`` clips a drawn value into the range the
    trained agent actually saw; a value outside it puts the policy's observation
    outside its trained range, which is a silent accuracy failure rather than an
    error (see ``counterfactual_env``'s module docstring).
    """

    name: str
    path: Path
    id_column: str
    parameters: Mapping[str, str]
    filters: Sequence[PoolFilter] = ()
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    integer_parameters: frozenset[str] = frozenset()
    #: Per-trial fit tables repeat one participant's parameters on every row.
    dedupe_by_participant: bool = False
    #: Built by a function rather than read straight from one CSV.
    builder: Optional[str] = None


#: CoAX: 1,133 fitted assignments over 330 participants. The column names are
#: already the strategy constructors' own keyword names, so no renaming is
#: needed. No ranges: ``COAX_PARAM_BOUNDS`` is the UI's slider range, not a
#: bound on the fitted population, and several fitted means sit outside it.
#:
#: Coverage: adult, forest_cover and wine_quality only -- mushrooms was never
#: fitted, so a mushrooms CoAX study falls back (with a warning) to the fitted
#: means it uses today.
COAX_POOL = PoolSpec(
    name="coax",
    path=HUMAN_DATA / "CoAX" / "coax_fitted_strategies.csv",
    id_column="Participant ID",
    filters=(
        PoolFilter("dataId", "dataId", canon_dataset),
        PoolFilter("xai_method", "expMethod", canon_plain),
        PoolFilter("xai_type", "XAIType", canon_coax_xai_type),
        PoolFilter("tested_w_xai", "Tested w/ XAI", canon_tested),
        PoolFilter("strategy", "Strategy", canon_coax_strategy),
    ),
    parameters={
        "sensitivity": "sensitivity",
        "k": "k",
        "retrieval_threshold": "retrieval_threshold",
        "scaling_factor": "scaling_factor",
    },
    integer_parameters=frozenset({"k"}),
)


#: CoXAM counterfactual: the full replay, 270 participants across both datasets
#: and all six condition x complexity cells (14-33 participants each).
#:
#: Deliberately not ``coxam_counterfactual_fit.csv``: that file is a
#: 50-participant subset of exactly this replay (see ``build_human_data.py``),
#: and several of its cells hold only one to three participants.
#:
#: Its three fitted columns land on the counterfactual environment's own
#: parameters, and the fitted ranges match the trained ranges in
#: ``counterfactual_env.DEFAULT_COGNITIVE_PARAMS`` almost exactly (kappa
#: -1.998..0.497 vs [-2.0, 0.5]; epsilon 0.051..0.499 vs [0.0, 0.5]; gamma
#: 0.0001..0.0200 vs [0.0, 0.02]), which is the strongest available evidence
#: that the mapping is the intended one. ``random_response_rate`` was not
#: fitted and keeps its default.
COXAM_COUNTERFACTUAL_POOL = PoolSpec(
    name="coxam_counterfactual",
    path=HUMAN_DATA / "CoXAM" / "coxam_counterfactual_replay.csv",
    id_column="Participant Id",
    filters=(
        PoolFilter("dataId", "dataId", canon_dataset),
        PoolFilter("condition", "condition", canon_coxam_condition),
        PoolFilter("complexity", "complexity", canon_plain),
    ),
    parameters={
        "Retrieval threshold, κ": "memory_recall_threshold",
        "Margin, ε": "counterfactual_overshoot_fraction",
        "Opportunity cost, γ": "time_penalty_weight",
    },
    ranges={
        "memory_recall_threshold": (-2.0, 0.5),
        "counterfactual_overshoot_fraction": (0.0, 0.5),
        "time_penalty_weight": (0.0, 0.02),
    },
    dedupe_by_participant=True,
)


#: CoXAM forward: assembled from three files by :func:`build_coxam_forward_pool`
#: rather than read from one, because the fits are split by dataset and by
#: explanation family. See that function for the column mapping and its limits.
COXAM_FORWARD_POOL = PoolSpec(
    name="coxam_forward",
    path=HUMAN_DATA / "CoXAM",
    id_column="pool_participant_id",
    filters=(
        PoolFilter("dataId", "dataId", canon_dataset),
        PoolFilter("condition", "condition", canon_coxam_condition),
        PoolFilter("complexity", "complexity", canon_plain),
    ),
    parameters={
        "memory_recall_threshold": "memory_recall_threshold",
        "decision_noise": "decision_noise",
        "opportunity_cost": "opportunity_cost",
    },
    # The ranges the loaded checkpoint was trained on (its run config.json),
    # not CombinedPolicyConfig's dataclass defaults -- no run used those.
    ranges={
        "memory_recall_threshold": (-1.0, 2.0),
        "decision_noise": (0.3, 0.7),
        "opportunity_cost": (0.0, 0.02),
    },
    builder="build_coxam_forward_pool",
)


#: Sim2Real: 46 participants, 11-12 per explanation property -- a near 1:1 match
#: for a study running 12 participants per condition, so a draw without
#: replacement is close to a permutation of the real population.
#:
#: This fit selected the *strategy* per participant as well as its parameters
#: (``sparse`` splits 6 sensitive_features / 3 attribution_sum / 3
#: salient_features), so in diverse mode the strategy varies within a condition
#: too, superseding the flat ``SIM2REAL_STRATEGY_BY_PROPERTY`` table.
SIM2REAL_POOL = PoolSpec(
    name="sim2real",
    path=HUMAN_DATA / "Sim2Real" / "sim2real_participant_fits.csv",
    id_column="participant_id",
    filters=(PoolFilter("exp_property", "exp_property", canon_plain),),
    parameters={
        "strategy": "strategy",
        "aggregation": "aggregation",
        "confidence_scale": "confidence_scale",
        "confidence_intercept": "confidence_intercept",
        "comparison_scale": "comparison_scale",
        "comparison_intercept": "comparison_intercept",
        "comparison_C": "comparison_C",
        "max_features_attended": "max_features_attended",
        "guess_bias": "guess_bias",
        "lapse_rate": "lapse_rate",
        "use_exemplar_memory": "use_exemplar_memory",
        "memory_sensitivity": "memory_sensitivity",
        "memory_decay": "memory_decay",
        "retrieval_threshold": "retrieval_threshold",
        "k": "k",
        "sensitivity": "sensitivity",
        "always_attend_changed": "always_attend_changed",
    },
    integer_parameters=frozenset({"max_features_attended", "k"}),
)


POOLS: dict[str, PoolSpec] = {
    spec.name: spec
    for spec in (COAX_POOL, COXAM_COUNTERFACTUAL_POOL, COXAM_FORWARD_POOL, SIM2REAL_POOL)
}


# -- loading --------------------------------------------------------------


class PoolUnavailable(RuntimeError):
    """The fitted pool this run needs has not been built."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PoolUnavailable(
            f"Fitted parameter pool {path} is missing. Run assets/build_human_data.py "
            "to consolidate the human data, or pass parameter_pool= with your own "
            "fitted table."
        )
    return pd.read_csv(path, low_memory=False)


def build_coxam_forward_pool() -> pd.DataFrame:
    """One normalized table from the three CoXAM forward fits.

    The forward meta-policy exposes exactly three free parameters at evaluation
    time (``CombinedStrategyPolicyEnv._sample_episode_params``) while the fits
    are ACT-R/drift-diffusion shaped, so unlike every other pool this one needs
    a real mapping rather than a rename:

    * ``retrieval_threshold`` -> ``memory_recall_threshold``. Same construct (an
      ACT-R log-odds retrieval cutoff), fitted -2.0..1.5 against a trained
      -1.0..2.0, so it is clipped.
    * ``chi_value`` -> ``opportunity_cost``. Same construct (a per-second time
      penalty) and the fitted range is *exactly* the trained [0, 0.02]. Only the
      mushrooms fit carries it; the wine_quality fits swept ``compute_sf``
      instead, so wine participants leave ``opportunity_cost`` unset and the
      runner keeps its default.
    * ``ddm_s`` -> ``decision_noise``. The weakest of the three: both are the
      noise scale of the decision process, but the fitted 0.2..1.5 is wider than
      the trained 0.3..0.7, so clipping compresses the tails. Kept because
      dropping it would leave two thirds of the parameters shared across
      participants; revisit with a rank-mapping if the compression matters.

    ``T_enc``/``T_op``/``latency_factor``/``lapse``/``ddm_a``/``compute_sf`` have
    no counterpart the meta-policy reads and are deliberately dropped.
    """
    frames: list[pd.DataFrame] = []

    for file_name, condition in (
        ("coxam_forward_params_wine_quality_dt.csv", "decision_tree"),
        ("coxam_forward_params_wine_quality_lr.csv", "linear_regression"),
    ):
        raw = _read_csv(HUMAN_DATA / "CoXAM" / file_name)
        frames.append(
            pd.DataFrame(
                {
                    "pool_participant_id": raw["Participant Id"],
                    "dataId": raw["dataId"],
                    # The file *is* the condition: one fit per explanation family.
                    "condition": condition,
                    "complexity": raw["Complexity"],
                    "memory_recall_threshold": raw["retrieval_threshold"],
                    "decision_noise": raw["ddm_s"],
                    "opportunity_cost": np.nan,
                    "source_file": file_name,
                }
            )
        )

    mushrooms = _read_csv(HUMAN_DATA / "CoXAM" / "coxam_forward_fit_mushrooms.csv")
    # A per-trial dump: one participant's fitted parameters repeat on every one
    # of their ~500 rows.
    mushrooms = mushrooms.drop_duplicates(subset="Participant Id")
    frames.append(
        pd.DataFrame(
            {
                "pool_participant_id": mushrooms["Participant Id"],
                "dataId": mushrooms["dataId"],
                "condition": mushrooms["Condition"],
                "complexity": mushrooms["Complexity"],
                "memory_recall_threshold": mushrooms["retrieval_threshold"],
                "decision_noise": mushrooms["ddm_s"],
                "opportunity_cost": mushrooms["chi_value"],
                "source_file": "coxam_forward_fit_mushrooms.csv",
            }
        )
    )

    pool = pd.concat(frames, ignore_index=True)
    pool["condition"] = pool["condition"].map(canon_coxam_condition)
    pool["dataId"] = pool["dataId"].map(canon_dataset)
    pool["complexity"] = pool["complexity"].map(canon_plain)
    return pool


_BUILDERS: dict[str, Callable[[], pd.DataFrame]] = {
    "build_coxam_forward_pool": build_coxam_forward_pool,
}


@lru_cache(maxsize=None)
def load_pool(name: str) -> pd.DataFrame:
    """The fitted table for one pool, read once per process.

    Cached because ``/simulate`` runs repeatedly against a long-lived study
    session and two of these files are large (the counterfactual replay is
    10,758 x 47; the mushrooms forward dump is 71k rows).
    """
    spec = POOLS.get(name)
    if spec is None:
        raise KeyError(f"Unknown parameter pool {name!r}. Known: {sorted(POOLS)}.")
    if spec.builder:
        return _BUILDERS[spec.builder]()
    pool = _read_csv(spec.path)
    if spec.dedupe_by_participant:
        pool = pool.drop_duplicates(subset=spec.id_column)
    return pool


# -- matching and drawing -------------------------------------------------


def match_pool_rows(
    spec: PoolSpec,
    condition: Mapping[str, Any],
    *,
    pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Rows fitted to participants who were in this condition cell.

    Only the axes present in ``condition`` are applied, so a caller that cannot
    resolve one (a design with no complexity factor, say) matches across it
    instead of matching nothing.
    """
    frame = load_pool(spec.name) if pool is None else pool
    for pool_filter in spec.filters:
        if pool_filter.key not in condition:
            continue
        if pool_filter.column not in frame.columns:
            continue
        wanted = pool_filter.canon(condition[pool_filter.key])
        frame = frame[frame[pool_filter.column].map(pool_filter.canon) == wanted]
    return frame


def _coerce(spec: PoolSpec, name: str, value: Any) -> Any:
    """One CSV cell as the runner's own parameter value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        return text
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    number = float(value)
    low_high = spec.ranges.get(name)
    if low_high is not None:
        number = float(np.clip(number, low_high[0], low_high[1]))
    if name in spec.integer_parameters:
        return int(round(number))
    return number


def _row_parameters(spec: PoolSpec, row: pd.Series) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for column, name in spec.parameters.items():
        if column not in row.index:
            continue
        value = _coerce(spec, name, row[column])
        if value is not None:
            parameters[name] = value
    return parameters


@dataclass(frozen=True)
class ParticipantDraw:
    """One virtual participant's assignment from the pool."""

    participant_id: Any
    parameters: dict[str, Any]
    fitted_participant_id: Optional[Any]
    parameter_source: str
    pool_name: str

    @property
    def is_fallback(self) -> bool:
        """True only when nothing was drawn at all.

        A relaxed draw (``pool_relaxed:...``) is still a real fitted human, just
        not one from this exact cell, so it does not count as a fallback.
        """
        return self.parameter_source == "fitted_mean_fallback"


def match_with_relaxation(
    spec: PoolSpec,
    condition: Mapping[str, Any],
    *,
    relax: Sequence[str] = (),
    pool: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Match the exact cell, then widen along ``relax`` until something matches.

    Real fitted populations have holes a simulated design does not: CoXAM's
    wine_quality forward fits were run per explanation family, so a ``hybrid``
    participant has no cell of its own, and CoAX fitted three of its four
    datasets. Widening one named axis at a time keeps the closest available
    match -- a hybrid participant drawn from that dataset's DT and LR fits is a
    far better stand-in than the framework's global mean -- and reports which
    axes it had to give up, so the compromise is visible in the results table
    rather than hidden.

    Returns the matched rows and the axes that were dropped to get them.
    """
    matched = match_pool_rows(spec, condition, pool=pool)
    if not matched.empty:
        return matched, ()
    for depth in range(1, len(relax) + 1):
        dropped = tuple(relax[:depth])
        widened = {
            key: value for key, value in condition.items() if key not in dropped
        }
        matched = match_pool_rows(spec, widened, pool=pool)
        if not matched.empty:
            return matched, dropped
    return matched, tuple(relax)


def draw_participant_parameters(
    pool_name: str,
    *,
    condition: Mapping[str, Any],
    participants: Sequence[Any],
    seed: int = 0,
    replace: Optional[bool] = None,
    relax: Sequence[str] = (),
    pool: Optional[pd.DataFrame] = None,
) -> dict[Any, ParticipantDraw]:
    """Deal one fitted parameter row to each virtual participant.

    Dealt without replacement while the matched pool lasts, then reshuffled and
    dealt again -- so with a pool at least as large as the study, no two virtual
    participants are the same person, and with a smaller one the empirical
    distribution is still reproduced rather than resampled i.i.d. Pass
    ``replace=True`` for independent draws instead.

    ``seed`` makes the assignment reproducible; callers vary it per condition
    cell so two cells do not deal the same pool rows in the same order.

    ``relax`` names condition axes that may be given up, in order, when the
    exact cell is empty -- see :func:`match_with_relaxation`.

    An unmatched cell is not an error: the runner keeps whatever defaults it
    uses today, and the draw records ``parameter_source="fitted_mean_fallback"``
    so the results table says so instead of quietly looking like a real draw.
    """
    spec = POOLS[pool_name]
    matched, dropped = match_with_relaxation(spec, condition, relax=relax, pool=pool)
    source = "pool" if not dropped else "pool_relaxed:" + ",".join(dropped)
    if dropped and not matched.empty:
        warnings.warn(
            f"No {pool_name} participants were fitted for the exact condition "
            f"{dict(condition)!r}; drew from the pool with {list(dropped)} relaxed "
            "instead. The results table records this as "
            f"parameter_source={source!r}.",
            stacklevel=2,
        )

    if matched.empty:
        warnings.warn(
            f"No {pool_name} participants were fitted for condition "
            f"{dict(condition)!r}, so mode='diverse_participant' falls back to the "
            "framework's fitted-population defaults for these participants -- they "
            "will not differ from each other. Check the condition against the pool, "
            "or accept the fallback.",
            stacklevel=2,
        )
        return {
            participant: ParticipantDraw(
                participant_id=participant,
                parameters={},
                fitted_participant_id=None,
                parameter_source="fitted_mean_fallback",
                pool_name=pool_name,
            )
            for participant in participants
        }

    rng = np.random.default_rng(seed)
    positions: list[int] = []
    if replace:
        positions = list(rng.integers(0, len(matched), size=len(participants)))
    else:
        while len(positions) < len(participants):
            positions.extend(rng.permutation(len(matched)).tolist())
        positions = positions[: len(participants)]

    draws: dict[Any, ParticipantDraw] = {}
    for participant, position in zip(participants, positions):
        row = matched.iloc[int(position)]
        draws[participant] = ParticipantDraw(
            participant_id=participant,
            parameters=_row_parameters(spec, row),
            fitted_participant_id=(
                row[spec.id_column] if spec.id_column in row.index else None
            ),
            parameter_source=source,
            pool_name=pool_name,
        )
    return draws


def draws_to_frame(draws: Mapping[Any, ParticipantDraw]) -> pd.DataFrame:
    """The assignment as one row per virtual participant, for provenance.

    Stored on ``study.participant_parameters`` and written next to the results
    CSV, so a run can be audited (and a tutorial can plot the distribution it
    actually sampled) rather than taken on trust.
    """
    rows = []
    for draw in draws.values():
        row: dict[str, Any] = {
            "participantId": draw.participant_id,
            "pool": draw.pool_name,
            "fitted_participant_id": draw.fitted_participant_id,
            "parameter_source": draw.parameter_source,
        }
        row.update({f"sampled_{key}": value for key, value in draw.parameters.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def provenance_columns(draw: Optional[ParticipantDraw]) -> dict[str, Any]:
    """The provenance fields stamped onto every simulated result row."""
    if draw is None:
        return {}
    columns: dict[str, Any] = {
        "fitted_participant_id": draw.fitted_participant_id,
        "parameter_source": draw.parameter_source,
        "parameter_pool": draw.pool_name,
    }
    columns.update({f"sampled_{key}": value for key, value in draw.parameters.items()})
    return columns


#: Mode name that turns per-participant sampling on, everywhere it is checked.
DIVERSE_PARTICIPANT_MODE = "diverse_participant"


def is_diverse_mode(mode: Any) -> bool:
    """Whether this ``mode`` asks for per-participant fitted parameters."""
    return str(mode or "").strip().lower() == DIVERSE_PARTICIPANT_MODE


__all__ = [
    "COAX_POOL",
    "COXAM_COUNTERFACTUAL_POOL",
    "COXAM_FORWARD_POOL",
    "DIVERSE_PARTICIPANT_MODE",
    "HUMAN_DATA",
    "POOLS",
    "ParticipantDraw",
    "PoolFilter",
    "PoolSpec",
    "PoolUnavailable",
    "SIM2REAL_POOL",
    "build_coxam_forward_pool",
    "canon_coax_strategy",
    "canon_coax_xai_type",
    "canon_coxam_condition",
    "canon_dataset",
    "canon_tested",
    "draw_participant_parameters",
    "draws_to_frame",
    "is_diverse_mode",
    "load_pool",
    "match_pool_rows",
    "match_with_relaxation",
    "provenance_columns",
]

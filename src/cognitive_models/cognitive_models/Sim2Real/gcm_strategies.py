"""Attribution-sum preparation and cognitive model for Sim2Real trials.

The Sim2Real assets represent 12 raw Adult features with 67 encoded
dimensions.  Numeric features occupy one dimension each; categorical features
occupy one one-hot block each.  This module projects a 67D explanation to one
attribution per raw feature by selecting the active dimension in each block.

``counterfactuals_fake.csv`` is deliberately treated as a feature-value table,
not an explanation table.  Its ``a0_i`` ... ``a66_i`` columns encode the
changed instance.  For a categorical change, the changed 12D attribution
vector therefore substitutes the explanation weight at the newly active
one-hot dimension.

When ``deltas.csv`` is supplied, its ``answer`` column provides the supervised
comparison target: 1 means the AI feedback is "Higher" and 0 means "Lower".
This is a direction label, not an absolute Above/Below income class.  The
``originalPrediction`` column is checked against the explanation row's ``pred``.

The source CSVs are only read.  This module never rewrites or derives files in
the Sim2Real asset directories.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SIM2REAL_DATA_DIR = REPO_ROOT / "assets" / "ai_dataset" / "sim2real"
DEFAULT_SIM2REAL_EXPLANATIONS_DIR = (
    REPO_ROOT / "assets" / "explanations" / "xai_desiderata"
)


SIM2REAL_RAW_FEATURE_ORDER = (
    "age",
    "capital-gain",
    "capital-loss",
    "education",
    "hours-per-week",
    "marital",
    "native-country",
    "occupation",
    "race",
    "relationship",
    "sex",
    "workclass",
)

# The tuple order above follows the raw observations/UI.  The dimension order
# differs because all four numeric columns precede the categorical blocks.
SIM2REAL_FEATURE_DIMENSIONS = {
    "age": (0,),
    "capital-gain": (1,),
    "capital-loss": (2,),
    "education": tuple(range(4, 20)),
    "hours-per-week": (3,),
    "marital": tuple(range(20, 27)),
    "native-country": tuple(range(27, 30)),
    "occupation": tuple(range(30, 45)),
    "race": tuple(range(45, 50)),
    "relationship": tuple(range(50, 56)),
    "sex": tuple(range(56, 58)),
    "workclass": tuple(range(58, 67)),
}

SIM2REAL_NUMERIC_FEATURES = frozenset(
    ("age", "capital-gain", "capital-loss", "hours-per-week")
)


def _sigmoid(value: float) -> float:
    """Numerically stable scalar logistic function."""
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _validate_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(feature_names)
    unknown = [name for name in names if name not in SIM2REAL_FEATURE_DIMENSIONS]
    if unknown:
        raise ValueError(f"Unknown Sim2Real raw feature(s): {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must not contain duplicates")
    return names


@dataclass(frozen=True)
class ProjectedAttributionPair:
    """Original and changed attribution vectors in the 12-feature space."""

    instance_id: int
    qid: int
    exp_property: Optional[str]
    feature_names: tuple[str, ...]
    original_dimension_indices: tuple[int, ...]
    changed_dimension_indices: tuple[int, ...]
    original_dimension_names: tuple[str, ...]
    changed_dimension_names: tuple[str, ...]
    original_feature_values: tuple[object, ...]
    changed_feature_values: tuple[object, ...]
    original_attributions: np.ndarray
    changed_attributions: np.ndarray
    original_value_multipliers: np.ndarray
    changed_value_multipliers: np.ndarray
    changed_feature_names: tuple[str, ...]
    ai_prediction: int
    correct_increase_label: Optional[int]
    split: Optional[str]

    def to_frame(self) -> pd.DataFrame:
        """Return a human-readable row for each projected raw feature."""
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "original_dimension": self.original_dimension_indices,
                "changed_dimension": self.changed_dimension_indices,
                "original_dimension_name": self.original_dimension_names,
                "changed_dimension_name": self.changed_dimension_names,
                "original_value": self.original_feature_values,
                "changed_value": self.changed_feature_values,
                "original_attribution": self.original_attributions,
                "changed_attribution": self.changed_attributions,
                "is_changed": [
                    name in self.changed_feature_names for name in self.feature_names
                ],
            }
        )


@dataclass(frozen=True)
class AttributionSumComparison:
    """Output of comparing original and counterfactual attribution sums."""

    instance_id: int
    qid: int
    exp_property: Optional[str]
    original_attribution_sum: float
    changed_attribution_sum: float
    p_original_high_income: float
    p_new_high_income: float
    confidence_delta: float
    pre_lapse_probability_income_increases: float
    guess_probability_income_increases: float
    probability_income_increases: float
    # Diagnostic only. Memory never decides the response: it fills the changed
    # feature's missing weight, and the response still comes from the
    # attribution sums through ``confidence_delta = p_new - p_original``.
    memory_imputed_weight_change: Optional[float]
    retrieved_exemplar_count: int
    lapse_rate: float
    effective_lapse_rate: float
    changed_feature_visible: bool
    changed_feature_names: tuple[str, ...]
    attended_feature_names: tuple[str, ...]


class Sim2RealAttributionProjector:
    """Read Sim2Real tables and project 67D explanations to 12 features."""

    def __init__(
        self,
        values: pd.DataFrame,
        attributions: pd.DataFrame,
        counterfactuals: pd.DataFrame,
        *,
        metadata: Optional[pd.DataFrame] = None,
        deltas: Optional[pd.DataFrame] = None,
        app_id: str = "adult_sim2real",
        model_name: str = "synthetic_ai",
        exp_method: str = "lime",
    ):
        self.values = values.copy()
        self.attributions = attributions.copy()
        self.counterfactuals = counterfactuals.copy()
        self.metadata = None if metadata is None else metadata.copy()
        self.deltas = None if deltas is None else deltas.copy()
        self.app_id = app_id
        self.model_name = model_name
        self.exp_method = exp_method
        self._validate_tables()

    @classmethod
    def from_csv_files(
        cls,
        *,
        values_csv,
        attribution_csv,
        counterfactual_csv,
        metadata_csv=None,
        deltas_csv=None,
        app_id: str = "adult_sim2real",
        model_name: str = "synthetic_ai",
        exp_method: str = "lime",
    ) -> "Sim2RealAttributionProjector":
        """Load the three source tables without modifying them.

        When ``metadata_csv`` is omitted, ``metadata.csv`` beside
        ``values_csv`` is used when present.
        """
        values_path = Path(values_csv)
        if metadata_csv is None:
            candidate = values_path.with_name("metadata.csv")
            metadata_csv = candidate if candidate.is_file() else None
        metadata = None if metadata_csv is None else pd.read_csv(metadata_csv)
        if deltas_csv is None:
            candidate = Path(counterfactual_csv).with_name("deltas.csv")
            deltas_csv = candidate if candidate.is_file() else None
        deltas = None if deltas_csv is None else pd.read_csv(deltas_csv)
        return cls(
            pd.read_csv(values_path),
            pd.read_csv(attribution_csv),
            pd.read_csv(counterfactual_csv),
            metadata=metadata,
            deltas=deltas,
            app_id=app_id,
            model_name=model_name,
            exp_method=exp_method,
        )

    @classmethod
    def from_assets(
        cls,
        *,
        assets_root=None,
        app_id: str = "adult_sim2real",
        model_name: str = "synthetic_ai",
        exp_method: str = "lime",
    ) -> "Sim2RealAttributionProjector":
        """Load the repository's Sim2Real corpus and explanation assets.

        ``assets_root`` is the directory containing ``ai_dataset`` and
        ``explanations``. It defaults to this repository's ``assets`` folder.
        Every CSV is read into memory; no source asset is modified.
        """
        root = Path(assets_root) if assets_root is not None else REPO_ROOT / "assets"
        data_dir = root / "ai_dataset" / "sim2real"
        explanations_dir = root / "explanations" / "xai_desiderata"
        return cls.from_csv_files(
            values_csv=data_dir / "values.csv",
            attribution_csv=explanations_dir / "attribution.csv",
            counterfactual_csv=explanations_dir / "counterfactuals_fake.csv",
            metadata_csv=data_dir / "metadata.csv",
            deltas_csv=explanations_dir / "deltas.csv",
            app_id=app_id,
            model_name=model_name,
            exp_method=exp_method,
        )

    @property
    def instance_ids(self) -> tuple[int, ...]:
        frame = self._filter_app(self.values)
        return tuple(sorted(frame["instanceId"].astype(int).unique().tolist()))

    def project(
        self,
        instance_id: int,
        *,
        exp_property: Optional[str],
        feature_names: Sequence[str] = SIM2REAL_RAW_FEATURE_ORDER,
        normalize_by_i_max: bool = False,
    ) -> ProjectedAttributionPair:
        """Project one original/counterfactual trial into raw-feature space.

        ``exp_property=None`` selects the unoptimized LIME baseline.  Otherwise
        pass one of ``faithful``, ``robust``, ``sparse``, or ``sparse_robust``.
        Missing source attributions are preserved as ``NaN`` so the cognitive
        model can either reject them or explicitly replace them with zero.
        """
        selected_features = _validate_feature_names(feature_names)
        original_row = self._instance_row(self.values, instance_id)
        attribution_row = self._explanation_row(
            self.attributions, instance_id, exp_property
        )
        changed_row = self._explanation_row(
            self.counterfactuals, instance_id, exp_property
        )
        qid = self._qid_value(
            original_row["qid"], table="values", instance_id=instance_id
        )
        for table_name, row in (
            ("attributions", attribution_row),
            ("counterfactuals", changed_row),
        ):
            row_qid = self._qid_value(
                row["qid"], table=table_name, instance_id=instance_id
            )
            if row_qid != qid:
                raise ValueError(
                    f"instanceId={instance_id} has qid={qid} in values but "
                    f"qid={row_qid} in {table_name}"
                )

        original_encoded = np.asarray(
            [original_row[f"v{i}"] for i in range(67)], dtype=float
        )
        changed_encoded = np.asarray(
            [changed_row[f"a{i}_i"] for i in range(67)], dtype=float
        )
        explanation = np.asarray(
            [attribution_row[f"a{i}_i"] for i in range(67)], dtype=float
        )
        if normalize_by_i_max:
            i_max = float(attribution_row["i_max"])
            if not np.isfinite(i_max) or i_max == 0.0:
                raise ValueError(
                    f"instanceId={instance_id} has invalid attribution i_max={i_max}"
                )
            explanation = explanation / i_max

        original_dimensions = tuple(
            self._active_dimension(original_encoded, feature)
            for feature in selected_features
        )
        changed_dimensions = tuple(
            self._active_dimension(changed_encoded, feature)
            for feature in selected_features
        )
        original_values = tuple(
            self._feature_value(original_encoded, feature, dimension)
            for feature, dimension in zip(selected_features, original_dimensions)
        )
        changed_values = tuple(
            self._feature_value(changed_encoded, feature, dimension)
            for feature, dimension in zip(selected_features, changed_dimensions)
        )
        changed_features = tuple(
            feature
            for feature, original_dimension, changed_dimension, original_value, changed_value
            in zip(
                selected_features,
                original_dimensions,
                changed_dimensions,
                original_values,
                changed_values,
            )
            if original_dimension != changed_dimension
            or not self._same_value(original_value, changed_value)
        )

        delta_row = self._delta_row(instance_id) if self.deltas is not None else None
        correct_increase_label = None
        split = None
        if delta_row is not None:
            delta_qid = self._qid_value(
                delta_row["qid"], table="deltas", instance_id=instance_id
            )
            if delta_qid != qid:
                raise ValueError(
                    f"instanceId={instance_id} has qid={qid} in values but "
                    f"qid={delta_qid} in deltas"
                )
            correct_increase_label = self._binary_value(
                delta_row["answer"],
                column="answer",
                instance_id=instance_id,
            )
            original_prediction = self._binary_value(
                delta_row["originalPrediction"],
                column="originalPrediction",
                instance_id=instance_id,
            )
            explanation_prediction = self._binary_value(
                attribution_row["pred"],
                column="pred",
                instance_id=instance_id,
            )
            if original_prediction != explanation_prediction:
                raise ValueError(
                    f"instanceId={instance_id} has originalPrediction="
                    f"{original_prediction} in deltas but pred="
                    f"{explanation_prediction} in attributions"
                )
            if "split" in delta_row.index and pd.notna(delta_row["split"]):
                split = str(delta_row["split"])

        return ProjectedAttributionPair(
            instance_id=int(instance_id),
            qid=qid,
            exp_property=exp_property,
            feature_names=selected_features,
            original_dimension_indices=original_dimensions,
            changed_dimension_indices=changed_dimensions,
            original_dimension_names=tuple(
                self._dimension_name(index) for index in original_dimensions
            ),
            changed_dimension_names=tuple(
                self._dimension_name(index) for index in changed_dimensions
            ),
            original_feature_values=original_values,
            changed_feature_values=changed_values,
            original_attributions=explanation[list(original_dimensions)].copy(),
            changed_attributions=explanation[list(changed_dimensions)].copy(),
            original_value_multipliers=np.asarray(
                [
                    self._value_multiplier(feature, original_encoded[dimension], dimension)
                    for feature, dimension in zip(selected_features, original_dimensions)
                ],
                dtype=float,
            ),
            changed_value_multipliers=np.asarray(
                [
                    self._value_multiplier(feature, changed_encoded[dimension], dimension)
                    for feature, dimension in zip(selected_features, changed_dimensions)
                ],
                dtype=float,
            ),
            changed_feature_names=changed_features,
            ai_prediction=int(attribution_row["pred"]),
            correct_increase_label=correct_increase_label,
            split=split,
        )

    def project_all(
        self,
        *,
        exp_property: Optional[str],
        feature_names: Sequence[str] = SIM2REAL_RAW_FEATURE_ORDER,
        normalize_by_i_max: bool = False,
        instance_ids: Optional[Sequence[int]] = None,
    ) -> list[ProjectedAttributionPair]:
        """Project every available instance for one explanation condition."""
        selected_ids = self.instance_ids if instance_ids is None else instance_ids
        return [
            self.project(
                instance_id,
                exp_property=exp_property,
                feature_names=feature_names,
                normalize_by_i_max=normalize_by_i_max,
            )
            for instance_id in selected_ids
        ]

    def instance_ids_for_split(self, split: Optional[str] = None) -> tuple[int, ...]:
        """Return labeled instance IDs, optionally restricted to a raw split."""
        if self.deltas is None:
            raise ValueError("deltas.csv is required to select Sim2Real trial splits")
        rows = self._filter_app(self.deltas)
        if split is not None:
            if "split" not in rows.columns:
                raise ValueError("deltas table has no 'split' column")
            rows = rows[rows["split"].astype(str) == str(split)]
        return tuple(sorted(rows["instanceId"].astype(int).unique().tolist()))

    def increase_labels(
        self,
        instance_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return Higher=1/Lower=0 feedback labels in instance-ID order."""
        if self.deltas is None:
            raise ValueError("deltas.csv is required to load increase labels")
        selected_ids = self.instance_ids if instance_ids is None else instance_ids
        return np.asarray(
            [
                self._binary_value(
                    self._delta_row(instance_id)["answer"],
                    column="answer",
                    instance_id=instance_id,
                )
                for instance_id in selected_ids
            ],
            dtype=int,
        )

    def _validate_tables(self) -> None:
        value_columns = {
            "appId", "instanceId", "qid", *(f"v{i}" for i in range(67))
        }
        explanation_columns = {
            "appId",
            "modelName",
            "expMethod",
            "expProperty",
            "instanceId",
            "qid",
            "pred",
            "i_max",
            *(f"a{i}_i" for i in range(67)),
        }
        for name, frame, required in (
            ("values", self.values, value_columns),
            ("attributions", self.attributions, explanation_columns),
            ("counterfactuals", self.counterfactuals, explanation_columns),
        ):
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"{name} table is missing columns: {missing}")
        if self.deltas is not None:
            required = {
                "appId", "instanceId", "qid", "answer", "originalPrediction"
            }
            missing = sorted(required.difference(self.deltas.columns))
            if missing:
                raise ValueError(f"deltas table is missing columns: {missing}")
            rows = self._filter_app(self.deltas)
            if rows["instanceId"].duplicated().any():
                raise ValueError("deltas table has duplicate instanceId rows")
            value_ids = set(self.instance_ids)
            delta_ids = set(rows["instanceId"].astype(int).tolist())
            if value_ids != delta_ids:
                raise ValueError(
                    "values and deltas tables have different instance IDs: "
                    f"values_only={sorted(value_ids - delta_ids)}, "
                    f"deltas_only={sorted(delta_ids - value_ids)}"
                )

    def _filter_app(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[frame["appId"] == self.app_id]
        if "modelName" in result.columns:
            result = result[result["modelName"] == self.model_name]
        if "expMethod" in result.columns:
            result = result[result["expMethod"] == self.exp_method]
        return result

    def _instance_row(self, frame: pd.DataFrame, instance_id: int) -> pd.Series:
        rows = self._filter_app(frame)
        rows = rows[rows["instanceId"].astype(int) == int(instance_id)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one values row for instanceId={instance_id}; found {len(rows)}"
            )
        return rows.iloc[0]

    def _explanation_row(
        self,
        frame: pd.DataFrame,
        instance_id: int,
        exp_property: Optional[str],
    ) -> pd.Series:
        rows = self._filter_app(frame)
        rows = rows[rows["instanceId"].astype(int) == int(instance_id)]
        if exp_property is None:
            rows = rows[rows["expProperty"].isna()]
        else:
            rows = rows[
                rows["expProperty"].fillna("").astype(str).str.lower()
                == str(exp_property).lower()
            ]
        if len(rows) != 1:
            raise ValueError(
                "Expected one explanation row for "
                f"instanceId={instance_id}, exp_property={exp_property!r}; "
                f"found {len(rows)}"
            )
        return rows.iloc[0]

    def _delta_row(self, instance_id: int) -> pd.Series:
        rows = self._filter_app(self.deltas)
        rows = rows[rows["instanceId"].astype(int) == int(instance_id)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one delta row for instanceId={instance_id}; found {len(rows)}"
            )
        return rows.iloc[0]

    @staticmethod
    def _qid_value(value, *, table: str, instance_id: int) -> int:
        if pd.isna(value):
            raise ValueError(f"instanceId={instance_id} has missing qid in {table}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"instanceId={instance_id} has invalid qid={value!r} in {table}"
            ) from exc
        if not numeric.is_integer():
            raise ValueError(
                f"instanceId={instance_id} has non-integer qid={value!r} in {table}"
            )
        return int(numeric)

    @staticmethod
    def _binary_value(value, *, column: str, instance_id: int) -> int:
        if pd.isna(value):
            raise ValueError(f"instanceId={instance_id} has missing {column}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"instanceId={instance_id} has non-numeric {column}={value!r}"
            ) from exc
        if numeric not in (0.0, 1.0):
            raise ValueError(
                f"instanceId={instance_id} has non-binary {column}={value!r}"
            )
        return int(numeric)

    def _active_dimension(self, encoded: np.ndarray, feature: str) -> int:
        dimensions = SIM2REAL_FEATURE_DIMENSIONS[feature]
        if feature in SIM2REAL_NUMERIC_FEATURES:
            return dimensions[0]
        active = [index for index in dimensions if np.isclose(encoded[index], 1.0)]
        if len(active) != 1:
            raise ValueError(
                f"Feature {feature!r} must have exactly one active one-hot dimension; "
                f"found {active}"
            )
        return active[0]

    def _dimension_name(self, dimension: int) -> str:
        if self.metadata is None:
            return f"dimension_{dimension}"
        rows = self.metadata[self.metadata["appId"] == self.app_id]
        if len(rows) != 1 or f"a{dimension}" not in rows.columns:
            return f"dimension_{dimension}"
        return str(rows.iloc[0][f"a{dimension}"])

    def _feature_value(self, encoded: np.ndarray, feature: str, dimension: int):
        if feature in SIM2REAL_NUMERIC_FEATURES:
            return float(encoded[dimension])
        dimension_name = self._dimension_name(dimension)
        prefix = f"{feature}_"
        if dimension_name.startswith(prefix):
            return dimension_name[len(prefix):]
        return dimension_name

    def _value_multiplier(self, feature: str, value: float, dimension: int) -> float:
        if feature not in SIM2REAL_NUMERIC_FEATURES:
            return 1.0
        if self.metadata is None:
            raise ValueError(
                "metadata.csv is required for numeric value-weighted attribution sums"
            )
        rows = self.metadata[self.metadata["appId"] == self.app_id]
        minimum_column = f"v{dimension}_min"
        maximum_column = f"v{dimension}_max"
        if len(rows) != 1 or minimum_column not in rows or maximum_column not in rows:
            raise ValueError(f"Metadata has no bounds for numeric dimension {dimension}")
        minimum = float(rows.iloc[0][minimum_column])
        maximum = float(rows.iloc[0][maximum_column])
        if maximum == minimum:
            return 0.0
        return float((float(value) - minimum) / (maximum - minimum))

    @staticmethod
    def _same_value(left, right) -> bool:
        if isinstance(left, (float, int, np.floating, np.integer)) and isinstance(
            right, (float, int, np.floating, np.integer)
        ):
            return bool(np.isclose(float(left), float(right)))
        return left == right


class Sim2RealFittedAttributionSum:
    """Compare high-income confidence before and after the feature change.

    The default ``aggregation='attribution'`` follows the requested strategy
    exactly: sum the 12 selected explanation values and apply a logistic
    transform.  For numeric counterfactuals the dimension does not change, so
    their extracted attribution is necessarily unchanged.  Set
    ``aggregation='value_weighted'`` to multiply numeric weights by their
    metadata-normalized values, allowing numeric value changes to influence the
    confidence calculation.

    ``max_features_attended`` limits each sum to a shared subset of raw
    features. The changed feature is always included; remaining slots use the
    strongest absolute original contributions.

    ``confidence_delta`` is always ``p_new - p_original``.  Positive values
    therefore support an "income increases" response.

    When the changed feature has no visible attribution (both its original and
    changed weights are zero), ``lapse_rate`` mixes the strategy response with
    a guessing distribution:

    ``P(final) = (1 - lapse_rate) * P(strategy) + lapse_rate * sigmoid(guess_bias)``.

    Setting ``guess_bias=0`` recovers uniform binary lapses, i.e.
    ``lapse_rate / 2 + (1 - lapse_rate) * P(strategy)``.

    Set ``use_exemplar_memory=True`` to enable CoAX's exemplar memory.  It is
    gated exactly the way CoAX gates it: ``AttributionSum`` bypasses memory
    whenever an attribution explanation is present, and only retrieves when the
    explanation cannot supply the value, so retrieval here runs only on trials
    where the changed feature has no visible attribution.  On those trials the
    missing original and counterfactual weights are imputed with CoAX's
    similarity-weighted average over stored exemplars, and the imputed weights
    then feed the usual attribution sums.  Visible-attribution trials are
    untouched.

    Fitting stores single-feature exemplars the way CoAX's ``feedback`` does:
    one exemplar per attended feature per shown instance, keyed by that
    feature's value and labelled by the sign of its attribution.  Only features
    inside ``max_features_attended`` are keyed, so a feature that is never
    attended during training has no exemplars to retrieve and imputes to zero.
    Retrieval reuses CoAX's activation ``-memory_decay * ln(elapsed)``, the
    ``retrieval_threshold`` cutoff, and similarity
    ``exp(-memory_sensitivity * distance + activation)``.  Exemplars from the
    trial being predicted are skipped so a training trial cannot retrieve
    itself.
    """

    def __init__(
        self,
        *,
        confidence_scale: float = 1.0,
        confidence_intercept: float = 0.0,
        comparison_scale: float = 1.0,
        comparison_intercept: float = 0.0,
        comparison_C: float = 1e6,
        guess_bias: float = 0.0,
        lapse_rate: float = 0.0,
        use_exemplar_memory: bool = False,
        retrieval_threshold: Optional[float] = None,
        memory_sensitivity: float = 1.0,
        memory_decay: float = 0.5,
        aggregation: str = "attribution",
        max_features_attended: int = 12,
        missing_attributions: str = "raise",
    ):
        if aggregation not in {"attribution", "value_weighted"}:
            raise ValueError("aggregation must be 'attribution' or 'value_weighted'")
        if missing_attributions not in {"raise", "zero"}:
            raise ValueError("missing_attributions must be 'raise' or 'zero'")
        if not np.isfinite(comparison_C) or float(comparison_C) <= 0.0:
            raise ValueError("comparison_C must be a finite positive number")
        if not np.isfinite(guess_bias):
            raise ValueError("guess_bias must be finite")
        if not np.isfinite(lapse_rate) or not 0.0 <= float(lapse_rate) <= 1.0:
            raise ValueError("lapse_rate must be a finite number from 0 to 1")
        if retrieval_threshold is not None and not np.isfinite(retrieval_threshold):
            raise ValueError("retrieval_threshold must be finite or None")
        if not np.isfinite(memory_sensitivity) or float(memory_sensitivity) <= 0.0:
            raise ValueError("memory_sensitivity must be a finite positive number")
        if not np.isfinite(memory_decay) or float(memory_decay) < 0.0:
            raise ValueError("memory_decay must be a finite non-negative number")
        if (
            isinstance(max_features_attended, bool)
            or not isinstance(max_features_attended, (int, np.integer))
            or not 1 <= int(max_features_attended) <= len(SIM2REAL_RAW_FEATURE_ORDER)
        ):
            raise ValueError("max_features_attended must be an integer from 1 to 12")
        self.confidence_scale = float(confidence_scale)
        self.confidence_intercept = float(confidence_intercept)
        self.comparison_scale = float(comparison_scale)
        self.comparison_intercept = float(comparison_intercept)
        self.comparison_C = float(comparison_C)
        self.guess_bias = float(guess_bias)
        self.lapse_rate = float(lapse_rate)
        self.use_exemplar_memory = bool(use_exemplar_memory)
        self.retrieval_threshold = (
            None if retrieval_threshold is None else float(retrieval_threshold)
        )
        self.memory_sensitivity = float(memory_sensitivity)
        self.memory_decay = float(memory_decay)
        self.aggregation = aggregation
        self.max_features_attended = int(max_features_attended)
        self.missing_attributions = missing_attributions
        self.is_fitted = False
        self.fit_instance_ids_: tuple[int, ...] = ()
        self.training_accuracy_: Optional[float] = None
        self.comparison_estimator_ = None
        self.memory_exemplars_: tuple[dict, ...] = ()

    def compare(self, pair: ProjectedAttributionPair) -> AttributionSumComparison:
        """Calculate original/new confidence and probability of an increase."""
        original = self._contributions(pair, changed=False)
        changed = self._contributions(pair, changed=True)
        changed_feature_visible = self._changed_feature_visible(pair)
        imputed_weight_change: Optional[float] = None
        retrieved_count = 0
        if self.use_exemplar_memory and not changed_feature_visible:
            original, changed, imputed_weight_change, retrieved_count = self._impute_from_memory(
                pair, original, changed
            )
        attended_indices = self._attended_feature_indices(pair, original)
        original_sum = float(np.sum(original[list(attended_indices)]))
        changed_sum = float(np.sum(changed[list(attended_indices)]))
        p_original = _sigmoid(
            self.confidence_intercept + self.confidence_scale * original_sum
        )
        p_new = _sigmoid(
            self.confidence_intercept + self.confidence_scale * changed_sum
        )
        delta = p_new - p_original
        pre_lapse_probability = _sigmoid(
            self.comparison_intercept + self.comparison_scale * delta
        )
        guess_probability = _sigmoid(self.guess_bias)
        effective_lapse_rate = (
            0.0 if changed_feature_visible else self.lapse_rate
        )
        probability_increase = (
            (1.0 - effective_lapse_rate) * pre_lapse_probability
            + effective_lapse_rate * guess_probability
        )
        return AttributionSumComparison(
            instance_id=pair.instance_id,
            qid=pair.qid,
            exp_property=pair.exp_property,
            original_attribution_sum=original_sum,
            changed_attribution_sum=changed_sum,
            p_original_high_income=p_original,
            p_new_high_income=p_new,
            confidence_delta=delta,
            pre_lapse_probability_income_increases=pre_lapse_probability,
            guess_probability_income_increases=guess_probability,
            probability_income_increases=probability_increase,
            memory_imputed_weight_change=imputed_weight_change,
            retrieved_exemplar_count=retrieved_count,
            lapse_rate=self.lapse_rate,
            effective_lapse_rate=effective_lapse_rate,
            changed_feature_visible=changed_feature_visible,
            changed_feature_names=pair.changed_feature_names,
            attended_feature_names=tuple(
                pair.feature_names[index] for index in attended_indices
            ),
        )

    def compare_many(
        self, pairs: Iterable[ProjectedAttributionPair]
    ) -> list[AttributionSumComparison]:
        return [self.compare(pair) for pair in pairs]

    def fit(
        self,
        pairs: Sequence[ProjectedAttributionPair],
        increase_labels: Optional[Sequence[int]] = None,
    ) -> "Sim2RealFittedAttributionSum":
        """Fit the final increase-response logistic transform.

        ``increase_labels`` must use 1 for "Higher" and 0 for "Lower".  The
        high-income confidence transform remains controlled by
        ``confidence_scale`` and ``confidence_intercept``.
        """
        from sklearn.linear_model import LogisticRegression

        if increase_labels is None:
            embedded = [pair.correct_increase_label for pair in pairs]
            if any(label is None for label in embedded):
                raise ValueError(
                    "increase_labels were not passed and projected pairs do not "
                    "contain deltas.csv answer labels"
                )
            increase_labels = embedded
        labels = np.asarray(increase_labels, dtype=int).reshape(-1)
        if len(pairs) != labels.size:
            raise ValueError("pairs and increase_labels must have the same length")
        if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
            raise ValueError("increase_labels must contain both binary classes 0 and 1")
        self.memory_exemplars_ = (
            self._build_memory_exemplars(pairs) if self.use_exemplar_memory else ()
        )
        self.fit_instance_ids_ = tuple(pair.instance_id for pair in pairs)
        deltas = np.asarray(
            [self.compare(pair).confidence_delta for pair in pairs], dtype=float
        ).reshape(-1, 1)
        estimator = LogisticRegression(
            C=self.comparison_C,
            solver="lbfgs",
            max_iter=1000,
        )
        estimator.fit(deltas, labels)
        self.comparison_scale = float(estimator.coef_[0, 0])
        self.comparison_intercept = float(estimator.intercept_[0])
        self.comparison_estimator_ = estimator
        self.is_fitted = True
        fitted_probabilities = self.predict_proba(pairs)[:, 1]
        self.training_accuracy_ = float(
            np.mean((fitted_probabilities >= 0.5).astype(int) == labels)
        )
        return self

    def fit_from_projector(
        self,
        projector: Sim2RealAttributionProjector,
        *,
        exp_property: Optional[str],
        split: Optional[str] = "training",
        feature_names: Sequence[str] = SIM2REAL_RAW_FEATURE_ORDER,
        normalize_by_i_max: bool = False,
    ) -> "Sim2RealFittedAttributionSum":
        """Project labeled trials and fit Higher/Lower feedback in one call.

        ``split`` defaults to ``"training"`` to avoid fitting against held-out
        test answers. Pass ``"test"`` deliberately or ``None`` for all 39
        trials. The fit consumes ``deltas.csv.answer`` through the projected
        pairs and leaves every source table untouched.
        """
        instance_ids = projector.instance_ids_for_split(split)
        pairs = projector.project_all(
            exp_property=exp_property,
            feature_names=feature_names,
            normalize_by_i_max=normalize_by_i_max,
            instance_ids=instance_ids,
        )
        return self.fit(pairs)

    def fit_from_assets(
        self,
        *,
        exp_property: Optional[str],
        split: Optional[str] = "training",
        feature_names: Sequence[str] = SIM2REAL_RAW_FEATURE_ORDER,
        normalize_by_i_max: bool = False,
        assets_root=None,
    ) -> "Sim2RealFittedAttributionSum":
        """Load repository assets, project trials, and fit in one call."""
        projector = Sim2RealAttributionProjector.from_assets(
            assets_root=assets_root
        )
        return self.fit_from_projector(
            projector,
            exp_property=exp_property,
            split=split,
            feature_names=feature_names,
            normalize_by_i_max=normalize_by_i_max,
        )

    def predict_proba(
        self, pairs: Sequence[ProjectedAttributionPair]
    ) -> np.ndarray:
        increase = np.asarray(
            [self.compare(pair).probability_income_increases for pair in pairs],
            dtype=float,
        )
        return np.column_stack((1.0 - increase, increase))

    def predict(self, pairs: Sequence[ProjectedAttributionPair]) -> np.ndarray:
        return (self.predict_proba(pairs)[:, 1] >= 0.5).astype(int)

    def _contributions(
        self, pair: ProjectedAttributionPair, *, changed: bool
    ) -> np.ndarray:
        values = np.asarray(
            pair.changed_attributions if changed else pair.original_attributions,
            dtype=float,
        ).copy()
        missing = ~np.isfinite(values)
        if np.any(missing):
            if self.missing_attributions == "raise":
                missing_features = np.asarray(pair.feature_names, dtype=object)[missing]
                raise ValueError(
                    f"instanceId={pair.instance_id} has missing attributions for "
                    f"{missing_features.tolist()}"
                )
            values[missing] = 0.0
        if self.aggregation == "value_weighted":
            multipliers = (
                pair.changed_value_multipliers
                if changed
                else pair.original_value_multipliers
            )
            values *= multipliers
        return values

    def _changed_feature_visible(self, pair: ProjectedAttributionPair) -> bool:
        """Whether the counterfactual feature has any displayed nonzero weight."""
        changed_indices = [
            pair.feature_names.index(name) for name in pair.changed_feature_names
        ]
        if not changed_indices:
            return False
        weights = np.concatenate(
            (
                np.asarray(pair.original_attributions, dtype=float)[changed_indices],
                np.asarray(pair.changed_attributions, dtype=float)[changed_indices],
            )
        )
        missing = ~np.isfinite(weights)
        if np.any(missing):
            if self.missing_attributions == "raise":
                raise ValueError(
                    f"instanceId={pair.instance_id} has missing attributions for "
                    f"changed feature(s) {list(pair.changed_feature_names)}"
                )
            weights[missing] = 0.0
        return bool(np.any(~np.isclose(weights, 0.0, atol=1e-12)))

    def _build_memory_exemplars(
        self,
        pairs: Sequence[ProjectedAttributionPair],
    ) -> tuple[dict, ...]:
        """Store CoAX-style single-feature exemplars for every shown instance.

        CoAX's ``feedback`` keys one exemplar per attended feature and, in
        attribution mode, labels it by the sign of that feature's attribution.
        Each training trial shows an original and a counterfactual instance, so
        both are stored, giving retrieval a value-to-attribution mapping per
        feature.
        """
        exemplars = []
        time_stored = 0.0
        for pair in pairs:
            original = self._contributions(pair, changed=False)
            changed = self._contributions(pair, changed=True)
            attended = self._attended_feature_indices(pair, original)
            shown = (
                (original, pair.original_value_multipliers, pair.original_dimension_indices),
                (changed, pair.changed_value_multipliers, pair.changed_dimension_indices),
            )
            for contributions, multipliers, dimensions in shown:
                time_stored += 1.0
                for index in attended:
                    attribution = float(contributions[index])
                    exemplars.append(
                        {
                            "instance_id": pair.instance_id,
                            "qid": pair.qid,
                            "feature_index": int(index),
                            "dimension_index": int(dimensions[index]),
                            "feature_value": float(multipliers[index]),
                            "attribution": attribution,
                            "label": 1 if attribution >= 0.0 else 0,
                            "time_stored": time_stored,
                        }
                    )
        return tuple(exemplars)

    def _impute_from_memory(
        self,
        pair: ProjectedAttributionPair,
        original: np.ndarray,
        changed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Optional[float], int]:
        """Fill invisible changed-feature weights from retrieved exemplars.

        Returns the two contribution vectors with the changed feature replaced
        by its imputed original and counterfactual weights, the resulting
        attribution delta for that feature, and how many exemplars were
        retrieved.
        """
        if not self.memory_exemplars_:
            return original, changed, None, 0
        imputed_original = np.asarray(original, dtype=float).copy()
        imputed_changed = np.asarray(changed, dtype=float).copy()
        total_delta = 0.0
        retrieved_count = 0
        for name in pair.changed_feature_names:
            index = pair.feature_names.index(name)
            before, count_before = self._impute_attribution(
                index,
                int(pair.original_dimension_indices[index]),
                float(pair.original_value_multipliers[index]),
                exclude_instance_id=pair.instance_id,
            )
            after, count_after = self._impute_attribution(
                index,
                int(pair.changed_dimension_indices[index]),
                float(pair.changed_value_multipliers[index]),
                exclude_instance_id=pair.instance_id,
            )
            imputed_original[index] = before
            imputed_changed[index] = after
            total_delta += after - before
            retrieved_count += count_before + count_after
        return imputed_original, imputed_changed, total_delta, retrieved_count

    def _impute_attribution(
        self,
        feature_index: int,
        dimension_index: int,
        feature_value: float,
        *,
        exclude_instance_id: Optional[int] = None,
    ) -> tuple[float, int]:
        """CoAX's similarity-weighted importance imputation for one feature.

        Mirrors the no-explanation branch of ``AttributionSum.infer``: only
        exemplars keyed at ``feature_index`` contribute, activation decays with
        storage age, ``retrieval_threshold`` filters the rest, and stored
        attributions whose sign disagrees with their label are flipped.
        Retrieving nothing yields ``0.0``, i.e. the pre-memory behaviour.

        Distance follows the raw one-hot space CoAX works in: two different
        categories of the same feature occupy different dimensions and are a
        full unit apart, while values inside one dimension differ by their
        metadata-normalized amount.  Without this, every category of a
        categorical feature would look identical and could never produce an
        imputed change.
        """
        current_time = float(len(self.memory_exemplars_) + 1)
        weighted_sum = 0.0
        total_similarity = 0.0
        retrieved_count = 0
        for exemplar in self.memory_exemplars_:
            if exemplar["feature_index"] != feature_index:
                continue
            if (
                exclude_instance_id is not None
                and exemplar["instance_id"] == exclude_instance_id
            ):
                continue
            elapsed = max(1e-12, current_time - float(exemplar["time_stored"]))
            activation = -self.memory_decay * float(np.log(elapsed))
            if (
                self.retrieval_threshold is not None
                and activation < self.retrieval_threshold
            ):
                continue
            if int(exemplar["dimension_index"]) != dimension_index:
                distance = 1.0
            else:
                distance = min(
                    1.0, abs(feature_value - float(exemplar["feature_value"]))
                )
            similarity = float(
                np.exp(-self.memory_sensitivity * distance + activation)
            )
            attribution = float(exemplar["attribution"])
            label_sign = 1.0 if exemplar["label"] == 1 else -1.0
            if math.copysign(1.0, attribution) != label_sign:
                attribution = -attribution
            weighted_sum += similarity * attribution
            total_similarity += similarity
            retrieved_count += 1
        if total_similarity <= 1e-12:
            return 0.0, retrieved_count
        return weighted_sum / total_similarity, retrieved_count

    def _attended_feature_indices(
        self,
        pair: ProjectedAttributionPair,
        original_contributions: np.ndarray,
    ) -> tuple[int, ...]:
        """Choose a shared feature set for the original and changed sums.

        The counterfactual's changed feature is always attended. Remaining
        slots are filled by the largest absolute original contributions. The
        same indices are then used for both sums so attention switching cannot
        create an artificial confidence delta.
        """
        changed = {
            pair.feature_names.index(name) for name in pair.changed_feature_names
        }
        if len(changed) > self.max_features_attended:
            raise ValueError(
                "max_features_attended is smaller than the number of changed "
                f"features for instanceId={pair.instance_id}"
            )
        remaining = [
            index
            for index in np.argsort(
                -np.abs(original_contributions), kind="stable"
            ).tolist()
            if index not in changed
        ]
        selected = changed.union(
            remaining[: self.max_features_attended - len(changed)]
        )
        return tuple(sorted(selected))

# ---------------------------------------------------------------------------
# Exemplar-similarity strategies
#
# The attribution-sum model above reads the explanation as its evidence, so a
# trial whose changed feature carries no visible attribution gives it a
# confidence_delta of exactly zero and it answers at chance. The two strategies
# below get their evidence from the feature values instead, and share the same
# fit/compare surface so one grid and one BIC can compare all three.
# ---------------------------------------------------------------------------
_COAX_MODEL_FILE = (
    Path(__file__).resolve().parents[1] / "coax" / "coax_gcm_multiple_strategies.py"
)


def _load_coax():
    """CoAX's own GCM helpers, loaded rather than reimplemented.

    ``euclidean_distance`` ignores ``None`` dimensions, clips element-wise
    differences to [-1, 1] and normalizes by the square root of the valid
    dimension count; ``compute_similarity`` and ``normalize_label_strengths``
    complete the GCM. Re-deriving any of them would be a second, subtly
    different model.
    """
    spec = importlib.util.spec_from_file_location(
        "coax_gcm_multiple_strategies", _COAX_MODEL_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CoAX strategies from {_COAX_MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Label the GCM votes on: the AI's own prediction, as CoAX's ``feedback`` stores it.
HIGH_INCOME_LABEL = 1


class Sim2RealValueEncoder:
    """Turn a pair's displayed feature values into a vector a GCM can compare.

    ``*_value_multipliers`` cannot be used for this. It encodes a categorical
    feature as ``1.0`` whichever category is present, so 26 of the 29 test
    counterfactuals -- every one that changes a categorical, e.g. education
    ``Bachelors -> Preschool`` -- produce an *identical* original and changed
    vector, a zero delta, and a model at chance. Exactly the failure this port
    exists to remove.

    So numeric features keep their multiplier (already scaled to [0, 1]) and
    categorical features become integer codes. Integer codes work because
    CoAX's ``euclidean_distance`` clips each element-wise difference to
    [-1, 1]: two different categories are at least one code apart and so
    contribute exactly 1, two equal categories contribute 0. That is the
    standard overlap metric for nominal features, and it falls out of CoAX's
    own clipping rather than needing a second distance function.

    Codes come from the categories seen at fit time, sorted for determinism.
    A category never seen in training collapses onto a single "unseen" code:
    it is correctly *different* from every training category, though two
    distinct unseen categories are not distinguished from each other.
    """

    def __init__(self, categories: dict[int, list[str]], numeric: set[int]) -> None:
        self.categories = categories
        self.numeric = numeric

    @classmethod
    def from_pairs(
        cls, pairs: Sequence[ProjectedAttributionPair]
    ) -> "Sim2RealValueEncoder":
        observed: dict[int, set[str]] = {}
        numeric: set[int] = set()
        for pair in pairs:
            for values in (pair.original_feature_values, pair.changed_feature_values):
                for index, value in enumerate(values):
                    text = str(value)
                    try:
                        float(text)
                    except ValueError:
                        observed.setdefault(index, set()).add(text)
                    else:
                        numeric.add(index)
        return cls(
            categories={index: sorted(values) for index, values in observed.items()},
            numeric=numeric - set(observed),
        )

    def encode(
        self,
        feature_values: Sequence[object],
        multipliers: Sequence[float],
    ) -> list[float]:
        vector: list[float] = []
        for index, value in enumerate(feature_values):
            if index in self.numeric:
                vector.append(float(multipliers[index]))
                continue
            known = self.categories.get(index, [])
            text = str(value)
            vector.append(
                float(known.index(text)) if text in known else float(len(known))
            )
        return vector


class _Sim2RealGCMStrategy:
    """Shared machinery: exemplar memory, GCM confidence, counterfactual compare."""

    def __init__(
        self,
        *,
        k: int = 3,
        sensitivity: float = 10.0,
        memory_decay: float = 0.5,
        retrieval_threshold: Optional[float] = None,
        comparison_scale: float = 1.0,
        comparison_intercept: float = 0.0,
        comparison_C: float = 1e6,
        guess_bias: float = 0.0,
        lapse_rate: float = 0.0,
        always_attend_changed: bool = True,
        random_seed: int = 0,
    ) -> None:
        if int(k) < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(sensitivity) or float(sensitivity) <= 0.0:
            raise ValueError("sensitivity must be a finite positive number")
        if not np.isfinite(comparison_C) or float(comparison_C) <= 0.0:
            raise ValueError("comparison_C must be a finite positive number")
        if not 0.0 <= float(lapse_rate) <= 1.0:
            raise ValueError("lapse_rate must be between 0 and 1")

        self.k = int(k)
        self.sensitivity = float(sensitivity)
        self.memory_decay = float(memory_decay)
        self.retrieval_threshold = (
            None if retrieval_threshold is None else float(retrieval_threshold)
        )
        self.comparison_scale = float(comparison_scale)
        self.comparison_intercept = float(comparison_intercept)
        self.comparison_C = float(comparison_C)
        self.guess_bias = float(guess_bias)
        self.lapse_rate = float(lapse_rate)
        # Defaults to True, and the fitter only ever searched True. With it off,
        # a counterfactual on a feature outside the attended set masks
        # identically in both vectors, so the delta is zero and the model
        # answers 0.5 -- measured at 0.21 non-zero deltas against 0.86 with it
        # on. The UI names the changed feature, so attending it is what the
        # participant does.
        self.always_attend_changed = bool(always_attend_changed)
        self.random_seed = int(random_seed)
        self.encoder: Optional[Sim2RealValueEncoder] = None

        self._coax = _load_coax()
        self.memory = self._coax.MemoryBase(
            decay_param=self.memory_decay, retrieval_threshold=self.retrieval_threshold
        )
        self.exemplar_count_ = 0
        self.fit_instance_ids_: tuple[int, ...] = ()
        self.training_accuracy_: Optional[float] = None
        self.is_fitted = False

    # -- memory ------------------------------------------------------------

    def store_exemplars(self, pairs: Sequence[ProjectedAttributionPair]) -> None:
        """One exemplar per training instance, labelled by the AI's prediction.

        This is CoAX's ``feedback``: the participant is shown the AI's answer
        during training, so the stored label is the AI's, not the ground truth.
        Times are the instance's position in the sequence so that activation
        ``-decay * ln(elapsed)`` is well defined when queried afterwards.
        """
        self.memory = self._coax.MemoryBase(
            decay_param=self.memory_decay, retrieval_threshold=self.retrieval_threshold
        )
        for position, pair in enumerate(pairs):
            self.memory.store_exemplar(
                label_probs={int(pair.ai_prediction): 1.0},
                features=self._vector(pair, changed=False),
                explanation=list(np.asarray(pair.original_attributions, dtype=float)),
                time_stored=float(position),
            )
        self.exemplar_count_ = len(pairs)

    def _vector(self, pair: ProjectedAttributionPair, *, changed: bool) -> list[float]:
        """The instance as the GCM sees it, categoricals included."""
        if self.encoder is None:
            self.encoder = Sim2RealValueEncoder.from_pairs([pair])
        values = pair.changed_feature_values if changed else pair.original_feature_values
        multipliers = (
            pair.changed_value_multipliers if changed else pair.original_value_multipliers
        )
        return self.encoder.encode(values, multipliers)

    def _retrieve(self) -> list[tuple[dict, float]]:
        # Queried one step past the last stored exemplar, so every elapsed time
        # is >= 1 and no activation is produced by the 1e-12 floor.
        return self.memory.retrieve_exemplars(current_time=float(self.exemplar_count_))

    # -- attention ---------------------------------------------------------

    def _focus_indices(self, pair: ProjectedAttributionPair) -> list[int]:
        raise NotImplementedError

    def _changed_indices(self, pair: ProjectedAttributionPair) -> list[int]:
        names = list(pair.feature_names)
        return [
            names.index(name) for name in pair.changed_feature_names if name in names
        ]

    def _random_focus(self, n_features: int) -> list[int]:
        rng = np.random.default_rng(self.random_seed)
        return list(rng.permutation(n_features)[: max(1, self.k)])

    def _attended(self, pair: ProjectedAttributionPair) -> list[int]:
        focus = list(self._focus_indices(pair))
        if self.always_attend_changed:
            # Without this, a counterfactual on an unattended feature masks
            # identically in both vectors, delta is 0, and the model is back at
            # chance -- the failure this port exists to fix. The UI names the
            # changed feature, so attending it is what the participant does.
            for index in self._changed_indices(pair):
                if index not in focus:
                    focus.append(index)
        return sorted(set(focus))

    # -- confidence --------------------------------------------------------

    def _mask(self, vector: Sequence[float], focus: Sequence[int]) -> list:
        focus_set = set(int(index) for index in focus)
        return [
            float(value) if index in focus_set else None
            for index, value in enumerate(vector)
        ]

    def _exemplar_vector(self, exemplar: dict, focus: Sequence[int]) -> list:
        """Subclasses that mask exemplars by their own explanation override this."""
        return self._mask(exemplar["features"], focus)

    def confidence(
        self, vector: Sequence[float], focus: Sequence[int]
    ) -> tuple[float, int]:
        """P(high income) for one instance, plus how many exemplars were read."""
        stored = self._retrieve()
        if not stored:
            return 0.5, 0
        test_vector = np.array(self._mask(vector, focus), dtype=float)
        label_strengths: dict[Any, float] = {}
        for exemplar, activation in stored:
            distance = self._coax.euclidean_distance(
                test_vector, np.array(self._exemplar_vector(exemplar, focus), dtype=float)
            )
            if distance is None:
                continue
            similarity = self._coax.compute_similarity(
                distance, activation, self.sensitivity, 1.0
            )
            for label, probability in exemplar["label_probs"].items():
                label_strengths[label] = (
                    label_strengths.get(label, 0.0) + similarity * probability
                )
        probabilities = self._coax.normalize_label_strengths(label_strengths)
        if not probabilities:
            return 0.5, len(stored)
        return float(probabilities.get(HIGH_INCOME_LABEL, 0.0)), len(stored)

    # -- the counterfactual wrapper ---------------------------------------

    @staticmethod
    def _changed_feature_visible(pair: ProjectedAttributionPair) -> bool:
        names = list(pair.feature_names)
        attributions = np.asarray(pair.original_attributions, dtype=float)
        for name in pair.changed_feature_names:
            if name in names and float(attributions[names.index(name)]) != 0.0:
                return True
        return False

    def compare(self, pair: ProjectedAttributionPair) -> AttributionSumComparison:
        focus = self._attended(pair)
        p_original, retrieved = self.confidence(self._vector(pair, changed=False), focus)
        p_new, _ = self.confidence(self._vector(pair, changed=True), focus)
        delta = p_new - p_original

        pre_lapse = _sigmoid(self.comparison_intercept + self.comparison_scale * delta)
        guess_probability = _sigmoid(self.guess_bias)
        changed_visible = self._changed_feature_visible(pair)
        effective_lapse = 0.0 if changed_visible else self.lapse_rate
        probability = (
            1.0 - effective_lapse
        ) * pre_lapse + effective_lapse * guess_probability

        return AttributionSumComparison(
            instance_id=pair.instance_id,
            qid=pair.qid,
            exp_property=pair.exp_property,
            # These two carry the strategy's confidence rather than a sum of
            # attributions; the column names are kept so every downstream
            # reader of a comparison keeps working.
            original_attribution_sum=p_original,
            changed_attribution_sum=p_new,
            p_original_high_income=p_original,
            p_new_high_income=p_new,
            confidence_delta=delta,
            pre_lapse_probability_income_increases=pre_lapse,
            guess_probability_income_increases=guess_probability,
            probability_income_increases=probability,
            memory_imputed_weight_change=None,
            retrieved_exemplar_count=retrieved,
            lapse_rate=self.lapse_rate,
            effective_lapse_rate=effective_lapse,
            changed_feature_visible=changed_visible,
            changed_feature_names=pair.changed_feature_names,
            attended_feature_names=tuple(pair.feature_names[index] for index in focus),
        )

    def compare_many(
        self, pairs: Iterable[ProjectedAttributionPair]
    ) -> list[AttributionSumComparison]:
        return [self.compare(pair) for pair in pairs]

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        pairs: Sequence[ProjectedAttributionPair],
        increase_labels: Optional[Sequence[int]] = None,
    ) -> "_Sim2RealGCMStrategy":
        """Store the training exemplars, then learn the comparison transform.

        Identical in shape to ``Sim2RealFittedAttributionSum.fit``, so the two
        model families are fitted and scored the same way and their likelihoods
        are comparable.
        """
        from sklearn.linear_model import LogisticRegression

        if increase_labels is None:
            embedded = [pair.correct_increase_label for pair in pairs]
            if any(label is None for label in embedded):
                raise ValueError(
                    "increase_labels were not passed and projected pairs do not "
                    "contain deltas.csv answer labels"
                )
            increase_labels = embedded
        labels = np.asarray(increase_labels, dtype=int).reshape(-1)
        if len(pairs) != labels.size:
            raise ValueError("pairs and increase_labels must have the same length")
        if len(np.unique(labels)) != 2:
            raise ValueError("increase_labels must contain both binary classes 0 and 1")

        if self.encoder is None:
            self.encoder = Sim2RealValueEncoder.from_pairs(pairs)
        self.store_exemplars(pairs)
        self.fit_instance_ids_ = tuple(pair.instance_id for pair in pairs)

        deltas = np.asarray(
            [self.compare(pair).confidence_delta for pair in pairs], dtype=float
        ).reshape(-1, 1)
        estimator = LogisticRegression(
            C=self.comparison_C, solver="lbfgs", max_iter=1000
        )
        estimator.fit(deltas, labels)
        self.comparison_scale = float(estimator.coef_[0, 0])
        self.comparison_intercept = float(estimator.intercept_[0])
        self.is_fitted = True

        fitted = np.asarray(
            [self.compare(pair).probability_income_increases for pair in pairs]
        )
        self.training_accuracy_ = float(
            np.mean((fitted >= 0.5).astype(int) == labels)
        )
        return self

    def predict_proba(
        self, pairs: Sequence[ProjectedAttributionPair]
    ) -> np.ndarray:
        probabilities = np.asarray(
            [self.compare(pair).probability_income_increases for pair in pairs],
            dtype=float,
        )
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, pairs: Sequence[ProjectedAttributionPair]) -> np.ndarray:
        return (self.predict_proba(pairs)[:, 1] >= 0.5).astype(int)


class Sim2RealFittedSensitiveFeatures(_Sim2RealGCMStrategy):
    """Attend the features that best separate the AI's two classes in memory.

    CoAX's ``none``-condition strategy. It never reads the explanation, so an
    invisible attribution costs it nothing -- the original and counterfactual
    instances still differ in the changed feature's *value*.
    """

    def _focus_indices(self, pair: ProjectedAttributionPair) -> list[int]:
        n_features = len(pair.feature_names)
        active = [exemplar for exemplar, _activation in self._retrieve()]
        if not active:
            return self._random_focus(n_features)

        groups: dict[Any, list] = {}
        for exemplar in active:
            label = max(exemplar["label_probs"], key=exemplar["label_probs"].get)
            groups.setdefault(label, []).append(exemplar["features"])
        if len(groups) != 2:
            return self._random_focus(n_features)

        first, second = (np.asarray(group, dtype=float) for group in groups.values())
        if first.shape[0] <= 1 or second.shape[0] <= 1:
            return self._random_focus(n_features)

        variance_floor = 1e-8
        standard_error = np.sqrt(
            np.maximum(np.nanvar(first, axis=0, ddof=1), variance_floor) / first.shape[0]
            + np.maximum(np.nanvar(second, axis=0, ddof=1), variance_floor)
            / second.shape[0]
        )
        standard_error[standard_error == 0] = 1e-12
        t_statistics = np.abs(
            np.nanmean(first, axis=0) - np.nanmean(second, axis=0)
        ) / standard_error
        return list(np.argsort(t_statistics)[::-1][: max(1, self.k)])


class Sim2RealFittedSalientFeatures(_Sim2RealGCMStrategy):
    """Attend where the explanation points, then reason over values.

    Degrades gracefully as visibility falls: it still classifies by feature
    values, so it keeps working when the explanation is uninformative -- it
    just stops aiming attention usefully.
    """

    def __init__(self, *, rank_signed_attributions: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rank_signed_attributions = bool(rank_signed_attributions)

    def _rank(self, attributions: Sequence[float]) -> np.ndarray:
        values = np.asarray(attributions, dtype=float)
        # CoAX ranks signed values, which is right for importance (already
        # absolute) but drops strongly negative Sim2Real attributions.
        return values if self.rank_signed_attributions else np.abs(values)

    def _focus_indices(self, pair: ProjectedAttributionPair) -> list[int]:
        ranked = self._rank(pair.original_attributions)
        if not np.any(np.isfinite(ranked)) or float(np.nanmax(ranked)) == 0.0:
            # Nothing to aim at -- every attribution is zero, which is the
            # sparse condition. Attend everything rather than an arbitrary k.
            return list(range(len(pair.feature_names)))
        return list(np.argsort(ranked)[::-1][: max(1, self.k)])

    def _exemplar_vector(self, exemplar: dict, focus: Sequence[int]) -> list:
        """Mask each exemplar by *its own* explanation, as CoAX does."""
        explanation = exemplar.get("explanation")
        if explanation is None:
            return self._mask(exemplar["features"], focus)
        ranked = self._rank(explanation)
        if float(np.nanmax(ranked)) == 0.0:
            return self._mask(exemplar["features"], focus)
        exemplar_focus = np.argsort(ranked)[::-1][: max(1, self.k)]
        return self._mask(exemplar["features"], exemplar_focus)


__all__ = [
    "AttributionSumComparison",
    "DEFAULT_SIM2REAL_DATA_DIR",
    "DEFAULT_SIM2REAL_EXPLANATIONS_DIR",
    "HIGH_INCOME_LABEL",
    "ProjectedAttributionPair",
    "SIM2REAL_FEATURE_DIMENSIONS",
    "SIM2REAL_NUMERIC_FEATURES",
    "SIM2REAL_RAW_FEATURE_ORDER",
    "Sim2RealAttributionProjector",
    "Sim2RealFittedAttributionSum",
    "Sim2RealFittedSalientFeatures",
    "Sim2RealFittedSensitiveFeatures",
    "Sim2RealValueEncoder",
]

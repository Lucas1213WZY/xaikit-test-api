"""Build a database-ready CoAX/CoXAM human-model fitting store.

The importer deliberately uses only the Python standard library so it remains
usable when the scientific Python environment is unavailable. It does not
modify or relocate source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).with_name("human_model_fitting_schema.sql")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "assets"
    / "human_trials_and_cognitive_parameters"
    / "consolidated.sqlite3"
)

COAX_ROOT = Path("src/cognitive_models/coax")
COXAM_ROOT = Path("src/cognitive_models/CoXAM")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text.lower()


def scalar(value: Any) -> Any:
    text = normalize_text(value)
    if not text or text.lower() in {"nan", "na", "null"}:
        return None
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return int(number)
        if math.isfinite(number):
            return number
    except ValueError:
        pass
    return text


def as_float(value: Any) -> float | None:
    result = scalar(value)
    return float(result) if isinstance(result, (int, float)) else None


def as_binary(value: Any) -> int | None:
    text = normalize_key(value)
    if text in {"1", "true", "yes", "w/ xai", "with xai", "with"}:
        return 1
    if text in {"0", "false", "no", "w/o xai", "without xai", "without"}:
        return 0
    return None


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            yield row_number, row


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


class StoreBuilder:
    def __init__(
        self,
        repo_root: Path,
        output_path: Path,
        participant_secret: bytes | None = None,
    ):
        self.repo_root = repo_root.resolve()
        self.output_path = output_path.resolve()
        self.participant_secret = participant_secret
        self.conn: sqlite3.Connection | None = None
        self.source_files: dict[str, int] = {}
        self.datasets: dict[tuple[str, str], int] = {}
        self.ai_models: dict[tuple[int, str], int] = {}
        self.studies: dict[str, int] = {}
        self.participants: dict[tuple[int, str], int] = {}

    @property
    def db(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("Store is not open")
        return self.conn

    def path(self, relative_path: str | Path) -> Path:
        return self.repo_root / relative_path

    def build(self) -> dict[str, Any]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        self.conn = sqlite3.connect(self.output_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        try:
            self._load_datasets("coax")
            self._load_datasets("coxam")
            self._load_predictions("coax")
            self._load_predictions("coxam")
            self._load_coax_explanations()
            self._load_coxam_surrogates()
            self._load_human_trials()
            self._load_coax_fits()
            self._load_dataset_provenance()
            self._load_model_artifacts()
            self._record_expected_missing_outputs()
            self.db.commit()
            self.db.execute("PRAGMA optimize")
            return self.audit_summary()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self.db.close()
            self.conn = None

    def add_source_file(self, relative_path: str | Path, framework: str, role: str) -> int:
        relative = Path(relative_path).as_posix()
        if relative in self.source_files:
            return self.source_files[relative]
        path = self.path(relative)
        if not path.exists():
            raise FileNotFoundError(path)
        row_count = csv_row_count(path) if path.suffix.lower() == ".csv" else None
        cursor = self.db.execute(
            """
            INSERT INTO source_files(
                framework, role, relative_path, sha256, byte_size, row_count, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                framework,
                role,
                relative,
                sha256_file(path),
                path.stat().st_size,
                row_count,
                utc_now(),
            ),
        )
        source_file_id = int(cursor.lastrowid)
        self.source_files[relative] = source_file_id
        return source_file_id

    def issue(
        self,
        severity: str,
        code: str,
        framework: str | None,
        *,
        source_file_id: int | None = None,
        source_row_number: int | None = None,
        entity_type: str | None = None,
        entity_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO data_quality_issues(
                severity, issue_code, framework, source_file_id, source_row_number,
                entity_type, entity_key, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                severity,
                code,
                framework,
                source_file_id,
                source_row_number,
                entity_type,
                entity_key,
                json_text(details or {}),
            ),
        )

    def source_paths(self, framework: str) -> dict[str, Path]:
        if framework == "coax":
            base = COAX_ROOT / "data" / "datasets" / "standard set"
            return {
                "metadata": base / "metadata.csv",
                "values": base / "values.csv",
                "predictions": base / "none.csv",
            }
        base = COXAM_ROOT / "datasets"
        return {
            "metadata": base / "metadata.csv",
            "values": base / "values.csv",
            "predictions": base / "none.csv",
        }

    def _load_datasets(self, framework: str) -> None:
        paths = self.source_paths(framework)
        metadata_source_id = self.add_source_file(paths["metadata"], framework, "dataset_metadata")
        values_source_id = self.add_source_file(paths["values"], framework, "dataset_instances")
        values_file_sha256 = sha256_file(self.path(paths["values"]))

        metadata_rows: dict[str, dict[str, str]] = {}
        for _row_number, row in csv_rows(self.path(paths["metadata"])):
            data_id = normalize_key(row.get("appId"))
            metadata_rows[data_id] = row

        for data_id, row in metadata_rows.items():
            features = []
            for index in range(100):
                name = normalize_text(row.get(f"a{index}"))
                if not name:
                    continue
                options = []
                option_count = scalar(row.get(f"v{index}_options"))
                if isinstance(option_count, (int, float)):
                    for option_index in range(int(option_count)):
                        option = scalar(row.get(f"v{index}_{option_index}"))
                        if option is not None:
                            options.append(option)
                min_value = as_float(row.get(f"v{index}_min"))
                max_value = as_float(row.get(f"v{index}_max"))
                features.append(
                    {
                        "index": index,
                        "name": name,
                        "kind": "categorical" if options else "numeric",
                        "min": min_value,
                        "max": max_value,
                        "options": options,
                    }
                )

            schema_hash = hashlib.sha256(json_text(features).encode("utf-8")).hexdigest()
            snapshot_hash = hashlib.sha256(
                f"{schema_hash}:{values_file_sha256}".encode("utf-8")
            ).hexdigest()
            version_key = f"{framework}:{data_id}:{snapshot_hash[:12]}"
            cursor = self.db.execute(
                """
                INSERT INTO dataset_versions(
                    framework, data_id, version_key, feature_schema_sha256,
                    snapshot_sha256, metadata_source_file_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    framework,
                    data_id,
                    version_key,
                    schema_hash,
                    snapshot_hash,
                    metadata_source_id,
                    json_text({key: scalar(value) for key, value in row.items()}),
                ),
            )
            dataset_version_id = int(cursor.lastrowid)
            self.datasets[(framework, data_id)] = dataset_version_id
            for feature in features:
                self.db.execute(
                    """
                    INSERT INTO dataset_features(
                        dataset_version_id, feature_index, feature_name, feature_kind,
                        min_value, max_value, options_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_version_id,
                        feature["index"],
                        feature["name"],
                        feature["kind"],
                        feature["min"],
                        feature["max"],
                        json_text(feature["options"]) if feature["options"] else None,
                    ),
                )

        seen_instance_keys: set[tuple[int, str]] = set()
        for row_number, row in csv_rows(self.path(paths["values"])):
            data_id = normalize_key(row.get("appId"))
            dataset_version_id = self.datasets.get((framework, data_id))
            source_instance_id = normalize_key(row.get("instanceId"))
            if dataset_version_id is None:
                self.issue(
                    "error",
                    "instance_missing_dataset_metadata",
                    framework,
                    source_file_id=values_source_id,
                    source_row_number=row_number,
                    entity_type="instance",
                    entity_key=f"{data_id}:{source_instance_id}",
                )
                continue
            key = (dataset_version_id, source_instance_id)
            if key in seen_instance_keys:
                self.issue(
                    "error",
                    "duplicate_instance_key",
                    framework,
                    source_file_id=values_source_id,
                    source_row_number=row_number,
                    entity_type="instance",
                    entity_key=f"{data_id}:{source_instance_id}",
                )
                continue
            seen_instance_keys.add(key)
            feature_values = []
            for index in range(100):
                column = f"v{index}"
                if column not in row:
                    break
                value = scalar(row.get(column))
                if value is None:
                    break
                feature_values.append(value)
            self.db.execute(
                """
                INSERT INTO instances(
                    dataset_version_id, source_instance_id, target_value,
                    feature_values_json, source_file_id, source_row_number
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_version_id,
                    source_instance_id,
                    normalize_key(row.get("y")) or None,
                    json_text(feature_values),
                    values_source_id,
                    row_number,
                ),
            )

        for data_id, dataset_version_id in self.datasets.items():
            if data_id[0] != framework:
                continue
            count = self.db.execute(
                "SELECT COUNT(*) FROM instances WHERE dataset_version_id = ?",
                (dataset_version_id,),
            ).fetchone()[0]
            if count == 0:
                self.issue(
                    "warning",
                    "dataset_metadata_without_instances",
                    framework,
                    source_file_id=metadata_source_id,
                    entity_type="dataset_version",
                    entity_key=data_id[1],
                )

    def get_ai_model(self, dataset_version_id: int, model_name: str) -> int:
        key = (dataset_version_id, normalize_key(model_name))
        existing = self.ai_models.get(key)
        if existing is not None:
            return existing
        cursor = self.db.execute(
            """
            INSERT INTO ai_models(dataset_version_id, model_name, model_role)
            VALUES (?, ?, 'black_box')
            """,
            key,
        )
        ai_model_id = int(cursor.lastrowid)
        self.ai_models[key] = ai_model_id
        return ai_model_id

    def _load_predictions(self, framework: str) -> None:
        relative_path = self.source_paths(framework)["predictions"]
        source_file_id = self.add_source_file(relative_path, framework, "ai_predictions")
        for row_number, row in csv_rows(self.path(relative_path)):
            data_id = normalize_key(row.get("appId"))
            instance_id = normalize_key(row.get("instanceId"))
            dataset_version_id = self.datasets.get((framework, data_id))
            if dataset_version_id is None:
                self.issue(
                    "error",
                    "prediction_missing_dataset",
                    framework,
                    source_file_id=source_file_id,
                    source_row_number=row_number,
                    entity_type="prediction",
                    entity_key=f"{data_id}:{instance_id}",
                )
                continue
            instance_exists = self.db.execute(
                """
                SELECT 1 FROM instances
                WHERE dataset_version_id = ? AND source_instance_id = ?
                """,
                (dataset_version_id, instance_id),
            ).fetchone()
            if not instance_exists:
                self.issue(
                    "error",
                    "prediction_missing_instance",
                    framework,
                    source_file_id=source_file_id,
                    source_row_number=row_number,
                    entity_type="prediction",
                    entity_key=f"{data_id}:{instance_id}",
                )
                continue
            ai_model_id = self.get_ai_model(dataset_version_id, row.get("modelName", ""))
            self.db.execute(
                """
                INSERT INTO ai_predictions(
                    ai_model_id, dataset_version_id, source_instance_id, prediction,
                    normalization_max, source_file_id, source_row_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ai_model_id,
                    dataset_version_id,
                    instance_id,
                    normalize_key(row.get("pred")) or None,
                    as_float(row.get("i_max")),
                    source_file_id,
                    row_number,
                ),
            )

    def _load_coax_explanations(self) -> None:
        base = COAX_ROOT / "data" / "datasets" / "standard set"
        for explanation_type in ("importance", "attribution"):
            relative_path = base / f"{explanation_type}.csv"
            source_file_id = self.add_source_file(
                relative_path, "coax", f"{explanation_type}_explanations"
            )
            for row_number, row in csv_rows(self.path(relative_path)):
                data_id = normalize_key(row.get("appId"))
                instance_id = normalize_key(row.get("instanceId"))
                dataset_version_id = self.datasets.get(("coax", data_id))
                if dataset_version_id is None:
                    continue
                ai_model_id = self.get_ai_model(dataset_version_id, row.get("modelName", ""))
                values = []
                for index in range(100):
                    column = f"a{index}_i"
                    if column not in row:
                        break
                    value = scalar(row.get(column))
                    if value is None:
                        break
                    values.append(value)
                try:
                    self.db.execute(
                        """
                        INSERT INTO local_explanations(
                            ai_model_id, dataset_version_id, source_instance_id,
                            explanation_type, explanation_method, prediction,
                            normalization_max, payload_json, source_file_id, source_row_number
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ai_model_id,
                            dataset_version_id,
                            instance_id,
                            explanation_type,
                            normalize_key(row.get("expMethod")),
                            normalize_key(row.get("pred")) or None,
                            as_float(row.get("i_max")),
                            json_text(
                                {
                                    "feature_values": values,
                                    "intercept": scalar(row.get("intercept")),
                                }
                            ),
                            source_file_id,
                            row_number,
                        ),
                    )
                except sqlite3.IntegrityError:
                    self.issue(
                        "error",
                        "explanation_missing_instance_or_duplicate",
                        "coax",
                        source_file_id=source_file_id,
                        source_row_number=row_number,
                        entity_type="local_explanation",
                        entity_key=f"{data_id}:{instance_id}",
                    )

    def _load_coxam_surrogates(self) -> None:
        base = COXAM_ROOT / "datasets"
        for surrogate_type, filename, variant_column in (
            ("logistic_regression", "logistic_regression.csv", "variant"),
            ("decision_tree", "decision_tree.csv", "depth"),
        ):
            relative_path = base / filename
            source_file_id = self.add_source_file(
                relative_path, "coxam", f"{surrogate_type}_explanations"
            )
            for row_number, row in csv_rows(self.path(relative_path)):
                data_id = normalize_key(row.get("appId"))
                dataset_version_id = self.datasets.get(("coxam", data_id))
                if dataset_version_id is None:
                    self.issue(
                        "error",
                        "surrogate_missing_dataset",
                        "coxam",
                        source_file_id=source_file_id,
                        source_row_number=row_number,
                        entity_type="surrogate_explanation",
                        entity_key=data_id,
                    )
                    continue
                ai_model_id = self.get_ai_model(dataset_version_id, row.get("model", ""))
                payload = {
                    key: scalar(value)
                    for key, value in row.items()
                    if key not in {"appId", "model", variant_column, "fidelity"}
                }
                self.db.execute(
                    """
                    INSERT INTO surrogate_explanations(
                        ai_model_id, surrogate_type, variant, fidelity, payload_json,
                        source_file_id, source_row_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ai_model_id,
                        surrogate_type,
                        normalize_key(row.get(variant_column)),
                        as_float(row.get("fidelity")),
                        json_text(payload),
                        source_file_id,
                        row_number,
                    ),
                )

    def participant_key(self, study_key: str, raw_participant_id: str) -> tuple[str, str]:
        message = f"{study_key}|{raw_participant_id}".encode("utf-8")
        if self.participant_secret:
            return hmac.new(self.participant_secret, message, hashlib.sha256).hexdigest(), "hmac-sha256"
        return hashlib.sha256(message).hexdigest(), "sha256-unsalted"

    def get_participant(
        self, study_id: int, study_key: str, raw_participant_id: Any
    ) -> int | None:
        raw = normalize_text(raw_participant_id)
        if not raw:
            return None
        participant_key, method = self.participant_key(study_key, raw)
        cache_key = (study_id, participant_key)
        existing = self.participants.get(cache_key)
        if existing is not None:
            return existing
        cursor = self.db.execute(
            """
            INSERT INTO participants(study_id, participant_key, pseudonym_method)
            VALUES (?, ?, ?)
            """,
            (study_id, participant_key, method),
        )
        participant_id = int(cursor.lastrowid)
        self.participants[cache_key] = participant_id
        return participant_id

    def create_study(
        self,
        framework: str,
        study_key: str,
        study_name: str,
        source_file_id: int,
    ) -> int:
        cursor = self.db.execute(
            """
            INSERT INTO studies(framework, study_key, study_name, source_file_id)
            VALUES (?, ?, ?, ?)
            """,
            (framework, study_key, study_name, source_file_id),
        )
        study_id = int(cursor.lastrowid)
        self.studies[study_key] = study_id
        return study_id

    def _trial_links(
        self,
        framework: str,
        data_id: Any,
        instance_id: Any,
        model_name: Any,
    ) -> tuple[int | None, int | None, list[str]]:
        issues = []
        dataset_version_id = self.datasets.get((framework, normalize_key(data_id)))
        source_instance_id = normalize_key(instance_id)
        if dataset_version_id is None:
            issues.append("missing_dataset")
            return None, None, issues
        if not source_instance_id:
            issues.append("missing_instance_id")
            return dataset_version_id, None, issues
        instance_exists = self.db.execute(
            """
            SELECT 1 FROM instances
            WHERE dataset_version_id = ? AND source_instance_id = ?
            """,
            (dataset_version_id, source_instance_id),
        ).fetchone()
        if not instance_exists:
            issues.append("unmatched_instance")
            return dataset_version_id, None, issues
        ai_model_id = self.ai_models.get((dataset_version_id, normalize_key(model_name)))
        if ai_model_id is None:
            issues.append("missing_ai_model")
        return dataset_version_id, ai_model_id, issues

    def _prediction_issues(
        self,
        ai_model_id: int | None,
        source_instance_id: Any,
        observed_prediction: Any,
    ) -> list[str]:
        if ai_model_id is None or not normalize_key(source_instance_id):
            return []
        stored_row = self.db.execute(
            """
            SELECT prediction FROM ai_predictions
            WHERE ai_model_id = ? AND source_instance_id = ?
            """,
            (ai_model_id, normalize_key(source_instance_id)),
        ).fetchone()
        if stored_row is None:
            return ["missing_stored_prediction"]
        observed = normalize_key(observed_prediction)
        if observed and observed != normalize_key(stored_row[0]):
            return ["prediction_mismatch"]
        return []

    def _load_human_trials(self) -> None:
        self._load_coax_human_trials()
        self._load_coxam_human_trials()
        if not self.participant_secret:
            self.issue(
                "warning",
                "unsalted_participant_pseudonyms",
                "shared",
                entity_type="participants",
                details={
                    "recommendation": "Set HUMAN_FITTING_HMAC_KEY before production ingestion."
                },
            )

    def _load_coax_human_trials(self) -> None:
        relative_path = (
            COAX_ROOT
            / "data"
            / "user study results"
            / "3-datasets-jan-09-2026-trials.csv"
        )
        source_file_id = self.add_source_file(relative_path, "coax", "human_trials")
        study_key = "coax:3-datasets-jan-09-2026"
        study_id = self.create_study(
            "coax", study_key, "CoAX three-dataset user study", source_file_id
        )
        for row_number, row in csv_rows(self.path(relative_path)):
            participant_id = self.get_participant(
                study_id, study_key, row.get("Participant ID")
            )
            dataset_version_id, ai_model_id, issues = self._trial_links(
                "coax",
                row.get("appId"),
                row.get("Instance Index"),
                row.get("modelName"),
            )
            issues.extend(
                self._prediction_issues(
                    ai_model_id,
                    row.get("Instance Index"),
                    row.get("AI Prediction"),
                )
            )
            xai_type = normalize_key(row.get("XAIType"))
            if (
                ai_model_id is not None
                and xai_type in {"importance", "attribution"}
                and not self.db.execute(
                    """
                    SELECT 1 FROM local_explanations
                    WHERE ai_model_id = ?
                      AND source_instance_id = ?
                      AND explanation_type = ?
                      AND explanation_method = ?
                    """,
                    (
                        ai_model_id,
                        normalize_key(row.get("Instance Index")),
                        xai_type,
                        normalize_key(row.get("expMethod")),
                    ),
                ).fetchone()
            ):
                issues.append("missing_explanation")
            step = normalize_key(row.get("Step"))
            response = normalize_key(row.get("Response")) or None
            if participant_id is None:
                issues.append("missing_participant_id")
            if step == "infer" and response is None:
                issues.append("missing_human_response")
            fit_eligible = int(not issues and step == "infer" and response is not None)
            self._insert_trial(
                study_id=study_id,
                participant_id=participant_id,
                dataset_version_id=dataset_version_id,
                ai_model_id=ai_model_id,
                source_file_id=source_file_id,
                row_number=row_number,
                source_trial_index=row.get("Trial Index"),
                source_instance_id=row.get("Instance Index"),
                session=row.get("Session"),
                phase=row.get("trialType"),
                step=row.get("Step"),
                trial_type=row.get("trialType"),
                tested_with_xai=as_binary(row.get("Tested w/ XAI")),
                xai_type=row.get("XAIType"),
                explanation_method=row.get("expMethod"),
                condition_name=row.get("XAIType"),
                complexity=None,
                ai_prediction=row.get("AI Prediction"),
                human_response=row.get("Response"),
                is_correct=as_binary(row.get("Correct")),
                response_time=as_float(row.get("Timing")),
                counterfactual_response=None,
                fit_eligible=fit_eligible,
                issues=issues,
                framework="coax",
            )

    def _load_coxam_human_trials(self) -> None:
        relative_path = (
            COXAM_ROOT
            / "datasets"
            / "mushrooms and wine quality user data v0.1.csv"
        )
        source_file_id = self.add_source_file(relative_path, "coxam", "human_trials")
        study_key = "coxam:mushrooms-wine-v0.1"
        study_id = self.create_study(
            "coxam",
            study_key,
            "CoXAM mushrooms and wine-quality user study v0.1",
            source_file_id,
        )
        counterfactual_columns = [
            "Changed feature index",
            "Changed feature name",
            "Changed feature type",
            "Changed from",
            "Changed to",
            "Changed amount",
            "Changed amount (normalized)",
            "Absolute changed amount (normalized)",
            "DT prediction (CF)",
            "LR prediction (CF)",
            "AI prediction (CF)",
            "Changed prediction (DT)",
            "Changed prediction (LR)",
            "Changed explainer",
            "Changed AI prediction",
        ]
        for row_number, row in csv_rows(self.path(relative_path)):
            participant_id = self.get_participant(
                study_id, study_key, row.get("Participant Id")
            )
            dataset_version_id, ai_model_id, issues = self._trial_links(
                "coxam", row.get("AppId"), row.get("Instance Id"), row.get("Model")
            )
            issues.extend(
                self._prediction_issues(
                    ai_model_id,
                    row.get("Instance Id"),
                    row.get("AI prediction"),
                )
            )
            condition = normalize_key(row.get("Condition"))
            complexity = normalize_key(row.get("Complexity"))
            required_surrogates = []
            if condition in {"lr", "hybrid"}:
                required_surrogates.append(
                    ("logistic_regression", "sparse" if complexity == "low" else "dense")
                )
            if condition in {"dt", "hybrid"}:
                required_surrogates.append(
                    ("decision_tree", "2" if complexity == "low" else "3")
                )
            for surrogate_type, variant in required_surrogates:
                if ai_model_id is not None and not self.db.execute(
                    """
                    SELECT 1 FROM surrogate_explanations
                    WHERE ai_model_id = ? AND surrogate_type = ? AND variant = ?
                    """,
                    (ai_model_id, surrogate_type, variant),
                ).fetchone():
                    issues.append("missing_explanation")
            phase = normalize_key(row.get("Phase"))
            response = normalize_key(row.get("Response")) or None
            if participant_id is None:
                issues.append("missing_participant_id")
            if phase == "forward" and response is None:
                issues.append("missing_human_response")
            if phase == "counterfactual" and (
                not normalize_text(row.get("Changed feature index"))
                or not normalize_text(row.get("Changed amount"))
            ):
                issues.append("missing_counterfactual_response")
            fit_eligible = int(
                not issues
                and (
                    (phase == "forward" and response is not None)
                    or phase == "counterfactual"
                )
            )
            counterfactual_response = (
                {
                    column: scalar(row.get(column))
                    for column in counterfactual_columns
                }
                if phase == "counterfactual"
                else None
            )
            self._insert_trial(
                study_id=study_id,
                participant_id=participant_id,
                dataset_version_id=dataset_version_id,
                ai_model_id=ai_model_id,
                source_file_id=source_file_id,
                row_number=row_number,
                source_trial_index=row.get("Trial Index"),
                source_instance_id=row.get("Instance Id"),
                session=None,
                phase=row.get("Phase"),
                step="response",
                trial_type=row.get("Phase"),
                tested_with_xai=as_binary(row.get("Tested w/ XAI")),
                xai_type=row.get("XAIType"),
                explanation_method=row.get("XAIType"),
                condition_name=row.get("Condition"),
                complexity=row.get("Complexity"),
                ai_prediction=row.get("AI prediction"),
                human_response=row.get("Response"),
                is_correct=as_binary(row.get("Response==AI")),
                response_time=as_float(row.get("Time")),
                counterfactual_response=counterfactual_response,
                fit_eligible=fit_eligible,
                issues=issues,
                framework="coxam",
            )

    def _insert_trial(
        self,
        *,
        study_id: int,
        participant_id: int | None,
        dataset_version_id: int | None,
        ai_model_id: int | None,
        source_file_id: int,
        row_number: int,
        source_trial_index: Any,
        source_instance_id: Any,
        session: Any,
        phase: Any,
        step: Any,
        trial_type: Any,
        tested_with_xai: int | None,
        xai_type: Any,
        explanation_method: Any,
        condition_name: Any,
        complexity: Any,
        ai_prediction: Any,
        human_response: Any,
        is_correct: int | None,
        response_time: float | None,
        counterfactual_response: dict[str, Any] | None,
        fit_eligible: int,
        issues: list[str],
        framework: str,
    ) -> None:
        status = "ok" if not issues else "excluded:" + ",".join(sorted(set(issues)))
        self.db.execute(
            """
            INSERT INTO trials(
                study_id, participant_id, dataset_version_id, ai_model_id,
                source_file_id, source_row_number, source_trial_index,
                source_instance_id, session, phase, step, trial_type,
                tested_with_xai, xai_type, explanation_method, condition_name,
                complexity, ai_prediction, human_response, is_correct,
                response_time, counterfactual_response_json, fit_eligible, quality_status
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                study_id,
                participant_id,
                dataset_version_id,
                ai_model_id,
                source_file_id,
                row_number,
                normalize_key(source_trial_index) or None,
                normalize_key(source_instance_id) or None,
                normalize_text(session) or None,
                normalize_key(phase) or None,
                normalize_key(step) or None,
                normalize_key(trial_type) or None,
                tested_with_xai,
                normalize_key(xai_type) or None,
                normalize_key(explanation_method) or None,
                normalize_key(condition_name) or None,
                normalize_key(complexity) or None,
                normalize_key(ai_prediction) or None,
                normalize_key(human_response) or None,
                is_correct,
                response_time,
                json_text(counterfactual_response)
                if counterfactual_response is not None
                else None,
                fit_eligible,
                status,
            ),
        )
        for issue_code in sorted(set(issues)):
            self.issue(
                (
                    "error"
                    if issue_code
                    in {
                        "missing_dataset",
                        "unmatched_instance",
                        "prediction_mismatch",
                    }
                    else "warning"
                ),
                f"trial_{issue_code}",
                framework,
                source_file_id=source_file_id,
                source_row_number=row_number,
                entity_type="trial",
                entity_key=f"row:{row_number}",
            )

    def _load_coax_fits(self) -> None:
        relative_path = (
            COAX_ROOT
            / "results"
            / "pop_em_subset"
            / "refit_all_assignments_detlocal-Jan-10.csv"
        )
        source_file_id = self.add_source_file(
            relative_path, "coax", "fitted_cognitive_parameters"
        )
        study_key = "coax:3-datasets-jan-09-2026"
        study_id = self.studies[study_key]
        parameter_columns = [
            "sensitivity",
            "k",
            "retrieval_threshold",
            "scaling_factor",
        ]
        for row_number, row in csv_rows(self.path(relative_path)):
            participant_id = self.get_participant(
                study_id, study_key, row.get("Participant ID")
            )
            dataset_version_id = self.datasets.get(
                ("coax", normalize_key(row.get("appId")))
            )
            ai_model_id = (
                self.ai_models.get(
                    (dataset_version_id, normalize_key(row.get("modelName")))
                )
                if dataset_version_id is not None
                else None
            )
            cursor = self.db.execute(
                """
                INSERT INTO cognitive_fit_runs(
                    framework, study_id, participant_id, dataset_version_id,
                    ai_model_id, source_file_id, source_row_number, session,
                    xai_type, explanation_method, tested_with_xai, strategy,
                    fit_algorithm, assignment_probability, objective_name,
                    objective_value, metrics_json
                ) VALUES (
                    'coax', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NLL', ?, ?
                )
                """,
                (
                    study_id,
                    participant_id,
                    dataset_version_id,
                    ai_model_id,
                    source_file_id,
                    row_number,
                    normalize_text(row.get("Session")) or None,
                    normalize_key(row.get("XAIType")) or None,
                    normalize_key(row.get("expMethod")) or None,
                    as_binary(row.get("Tested w/ XAI")),
                    normalize_text(row.get("Strategy")) or None,
                    "population_em_refit_deterministic_local",
                    as_float(row.get("Assignment Prob")),
                    as_float(row.get("NLL")),
                    json_text(
                        {
                            "PAI": scalar(row.get("PAI")),
                            "MAI": scalar(row.get("MAI")),
                        }
                    ),
                ),
            )
            fit_run_id = int(cursor.lastrowid)
            for parameter_name in parameter_columns:
                value = scalar(row.get(parameter_name))
                if value is None:
                    continue
                self.db.execute(
                    """
                    INSERT INTO cognitive_parameters(
                        fit_run_id, parameter_name, numeric_value, text_value
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        fit_run_id,
                        parameter_name,
                        float(value) if isinstance(value, (int, float)) else None,
                        None if isinstance(value, (int, float)) else str(value),
                    ),
                )

    def _load_dataset_provenance(self) -> None:
        original_root = self.path("assets/original_datasets")
        allowed_suffixes = {".csv", ".gz", ".json", ".npy", ".pkl", ".py"}
        generator_relative = COXAM_ROOT / "dataset_generator" / "generate_datasets.py"
        generated_data_ids = {
            "breast_cancer",
            "wine_binary",
            "anes96_vote",
            "fair_affairs",
            "rand_health_visits",
            "star98_pass_rate",
            "banknote_authentication",
            "spambase",
            "pima_diabetes",
            "ionosphere",
            "blood_transfusion",
        }
        generator_source_id = self.add_source_file(
            generator_relative, "coxam", "dataset_generation_code"
        )

        for (framework, data_id), dataset_version_id in self.datasets.items():
            dataset_dir = original_root / data_id
            if dataset_dir.is_dir():
                artifact_paths = sorted(
                    path
                    for path in dataset_dir.rglob("*")
                    if path.is_file()
                    and "__pycache__" not in path.parts
                    and path.name != ".DS_Store"
                    and path.suffix.lower() in allowed_suffixes
                )
                for artifact_path in artifact_paths:
                    relative = artifact_path.relative_to(self.repo_root)
                    source_file_id = self.add_source_file(
                        relative, "shared", "original_dataset_artifact"
                    )
                    relative_inside_dataset = artifact_path.relative_to(dataset_dir)
                    if relative_inside_dataset.parts[0] in {
                        "train",
                        "dev",
                        "test",
                        "base",
                    }:
                        provenance_role = "prepared_split"
                    elif artifact_path.name == "load.py":
                        provenance_role = "dataset_loader"
                    elif artifact_path.name.startswith("metadata"):
                        provenance_role = "dataset_metadata_artifact"
                    else:
                        provenance_role = "raw_or_preprocessed_dataset"
                    self.db.execute(
                        """
                        INSERT INTO dataset_provenance(
                            dataset_version_id, source_file_id, provenance_role, link_status
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            dataset_version_id,
                            source_file_id,
                            provenance_role,
                            "directory_name_match_unverified",
                        ),
                    )
            else:
                generated_source = (
                    framework == "coxam" and data_id in generated_data_ids
                )
                self.issue(
                    "info" if generated_source else "warning",
                    (
                        "original_dataset_local_copy_not_found"
                        if generated_source
                        else "original_dataset_directory_not_found"
                    ),
                    framework,
                    entity_type="dataset_version",
                    entity_key=f"{framework}:{data_id}",
                )

            if framework == "coxam" and data_id in generated_data_ids:
                self.db.execute(
                    """
                    INSERT INTO dataset_provenance(
                        dataset_version_id, source_file_id, provenance_role, link_status
                    ) VALUES (?, ?, 'dataset_generation_code', 'code_defined_source')
                    """,
                    (dataset_version_id, generator_source_id),
                )

    def _load_model_artifacts(self) -> None:
        artifact_paths = sorted(
            self.path("assets/model_weights").rglob("*_model_weights.pth")
        )
        artifact_paths.extend(
            sorted(self.path(COXAM_ROOT / "outputs").glob("**/models/*.zip"))
        )
        ai_model_lookup: dict[tuple[str, str, str], int] = {}
        for ai_model_id, framework, data_id, model_name in self.db.execute(
            """
            SELECT am.ai_model_id, dv.framework, dv.data_id, am.model_name
            FROM ai_models AS am
            JOIN dataset_versions AS dv ON dv.dataset_version_id = am.dataset_version_id
            """
        ):
            ai_model_lookup[(framework, data_id, model_name)] = ai_model_id

        linked_ai_model_ids: set[int] = set()
        for path in artifact_paths:
            relative = path.relative_to(self.repo_root).as_posix()
            parts = path.parts
            if "model_weights" in parts:
                index = parts.index("model_weights")
                model_name = parts[index + 1]
                framework = parts[index + 2]
                filename = path.name
                data_id = filename.removesuffix("_model_weights.pth")
                ai_model_id = ai_model_lookup.get((framework, data_id, model_name))
                artifact_kind = "ai_model_weights"
                link_status = "filename_match_unverified" if ai_model_id else "unlinked"
                if ai_model_id:
                    linked_ai_model_ids.add(ai_model_id)
                metadata = {
                    "model_name": model_name,
                    "data_id_candidate": data_id,
                }
            else:
                framework = "coxam"
                ai_model_id = None
                artifact_kind = "cognitive_policy_weights"
                link_status = "run_directory"
                run_dir = path.parent.parent
                config_path = run_dir / "config.json"
                summary_path = run_dir / "metrics" / "training_summary.json"
                metadata = {
                    "run_directory": run_dir.relative_to(self.repo_root).as_posix(),
                    "config_path": (
                        config_path.relative_to(self.repo_root).as_posix()
                        if config_path.exists()
                        else None
                    ),
                    "training_summary_path": (
                        summary_path.relative_to(self.repo_root).as_posix()
                        if summary_path.exists()
                        else None
                    ),
                }
            self.db.execute(
                """
                INSERT INTO model_artifacts(
                    framework, artifact_kind, ai_model_id, relative_path, sha256,
                    byte_size, link_status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    framework,
                    artifact_kind,
                    ai_model_id,
                    relative,
                    sha256_file(path),
                    path.stat().st_size,
                    link_status,
                    json_text(metadata),
                ),
            )

        for ai_model_id, framework, data_id, model_name in self.db.execute(
            """
            SELECT am.ai_model_id, dv.framework, dv.data_id, am.model_name
            FROM ai_models AS am
            JOIN dataset_versions AS dv ON dv.dataset_version_id = am.dataset_version_id
            """
        ):
            if ai_model_id not in linked_ai_model_ids:
                self.issue(
                    "warning",
                    "ai_model_weights_not_found",
                    framework,
                    entity_type="ai_model",
                    entity_key=f"{framework}:{data_id}:{model_name}",
                    details={
                        "expected_path": (
                            f"assets/model_weights/{model_name}/{framework}/"
                            f"{data_id}_model_weights.pth"
                        )
                    },
                )

    def _record_expected_missing_outputs(self) -> None:
        missing_paths = [
            COXAM_ROOT / "datasets" / "combined.csv",
            COXAM_ROOT / "participant_parameters_fit_lr_v0.1.csv",
            COXAM_ROOT / "participant_parameters_fit_dt_v0.1.csv",
            COXAM_ROOT / "outputs" / "rl_fit_trials.csv",
            COXAM_ROOT / "outputs" / "rl_fit_trials_round2.csv",
        ]
        for relative_path in missing_paths:
            if not self.path(relative_path).exists():
                self.issue(
                    "warning",
                    "referenced_fitting_output_missing",
                    "coxam",
                    entity_type="expected_file",
                    entity_key=relative_path.as_posix(),
                )

    def audit_summary(self) -> dict[str, Any]:
        counts = {}
        for table in (
            "source_files",
            "dataset_versions",
            "dataset_features",
            "dataset_provenance",
            "instances",
            "ai_models",
            "ai_predictions",
            "local_explanations",
            "surrogate_explanations",
            "studies",
            "participants",
            "trials",
            "cognitive_fit_runs",
            "cognitive_parameters",
            "model_artifacts",
            "data_quality_issues",
        ):
            counts[table] = self.db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        fit_ready = dict(
            self.db.execute(
                """
                SELECT s.framework, COUNT(*)
                FROM trials AS t
                JOIN studies AS s ON s.study_id = t.study_id
                WHERE t.fit_eligible = 1
                GROUP BY s.framework
                """
            ).fetchall()
        )
        issues = {
            code: count
            for code, count in self.db.execute(
                """
                SELECT issue_code, COUNT(*)
                FROM data_quality_issues
                GROUP BY issue_code
                ORDER BY issue_code
                """
            ).fetchall()
        }
        return {
            "generated_at": utc_now(),
            "database_path": str(self.output_path),
            "counts": counts,
            "fit_ready_trials_by_framework": fit_ready,
            "quality_issues": issues,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized SQLite store for CoAX/CoXAM human-model fitting."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to the root containing this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination SQLite database. An existing file at this exact path is replaced.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable audit summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secret_value = os.environ.get("HUMAN_FITTING_HMAC_KEY")
    builder = StoreBuilder(
        repo_root=args.repo_root,
        output_path=args.output,
        participant_secret=secret_value.encode("utf-8") if secret_value else None,
    )
    summary = builder.build()
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

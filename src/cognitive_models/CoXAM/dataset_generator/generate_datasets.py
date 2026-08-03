from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree
from xgboost import XGBClassifier


RANDOM_STATE = 123
MAX_ROWS_PER_DATASET = 400
N_FEATURES = 6
MODEL_NAMES = ("mlp", "xgboost")


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class DatasetSpec:
    app_id: str
    display_name: str
    source: str
    openml_names: tuple[str, ...] = ()
    openml_data_ids: tuple[int, ...] = ()
    sklearn_loader: Callable[[], tuple[pd.DataFrame, pd.Series]] | None = None
    positive_label: str | int | float | None = None


@dataclass(frozen=True)
class PreparedDataset:
    app_id: str
    display_name: str
    X_raw: pd.DataFrame
    X_model: np.ndarray
    y: np.ndarray
    feature_metadata: list[dict]
    target_labels: tuple[str, str]


def load_sklearn_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    target = data.target.map(lambda value: str(data.target_names[int(value)]))
    return data.data, target


def load_sklearn_wine_binary() -> tuple[pd.DataFrame, pd.Series]:
    data = load_wine(as_frame=True)
    frame = data.frame[data.frame["target"].isin([0, 1])].copy()
    target = frame.pop("target").map(lambda value: str(data.target_names[int(value)]))
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def load_statsmodels_anes96_vote() -> tuple[pd.DataFrame, pd.Series]:
    frame = sm.datasets.anes96.load_pandas().data.copy()
    target = frame.pop("vote").map(lambda value: "dole" if int(value) == 1 else "clinton")
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def load_statsmodels_fair_affairs() -> tuple[pd.DataFrame, pd.Series]:
    frame = sm.datasets.fair.load_pandas().data.copy()
    target = frame.pop("affairs").map(lambda value: "affair" if float(value) > 0.0 else "no_affair")
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def load_statsmodels_rand_visits() -> tuple[pd.DataFrame, pd.Series]:
    frame = sm.datasets.randhie.load_pandas().data.copy()
    target = frame.pop("mdvis").map(lambda value: "visited_doctor" if float(value) > 0.0 else "no_visit")
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def load_statsmodels_star98_pass_rate() -> tuple[pd.DataFrame, pd.Series]:
    frame = sm.datasets.star98.load_pandas().data.copy()
    pass_rate = frame["NABOVE"] / (frame["NABOVE"] + frame["NBELOW"])
    target = pass_rate.map(lambda value: "high_pass_rate" if value >= pass_rate.median() else "low_pass_rate")
    frame = frame.drop(columns=["NABOVE", "NBELOW"])
    return frame.reset_index(drop=True), target.reset_index(drop=True)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "breast_cancer": DatasetSpec(
        app_id="breast_cancer",
        display_name="Breast Cancer Wisconsin",
        source="sklearn",
        sklearn_loader=load_sklearn_breast_cancer,
    ),
    "wine_binary": DatasetSpec(
        app_id="wine_binary",
        display_name="Wine Cultivar 0 vs 1",
        source="sklearn",
        sklearn_loader=load_sklearn_wine_binary,
    ),
    "anes96_vote": DatasetSpec(
        app_id="anes96_vote",
        display_name="ANES 1996 Vote",
        source="statsmodels",
        sklearn_loader=load_statsmodels_anes96_vote,
    ),
    "fair_affairs": DatasetSpec(
        app_id="fair_affairs",
        display_name="Fair Affairs Survey",
        source="statsmodels",
        sklearn_loader=load_statsmodels_fair_affairs,
    ),
    "rand_health_visits": DatasetSpec(
        app_id="rand_health_visits",
        display_name="RAND Health Medical Visits",
        source="statsmodels",
        sklearn_loader=load_statsmodels_rand_visits,
    ),
    "star98_pass_rate": DatasetSpec(
        app_id="star98_pass_rate",
        display_name="STAR98 School Pass Rate",
        source="statsmodels",
        sklearn_loader=load_statsmodels_star98_pass_rate,
    ),
    "banknote_authentication": DatasetSpec(
        app_id="banknote_authentication",
        display_name="Banknote Authentication",
        source="openml",
        openml_names=("banknote-authentication", "banknote_authentication"),
        openml_data_ids=(1462,),
    ),
    "spambase": DatasetSpec(
        app_id="spambase",
        display_name="Spambase",
        source="openml",
        openml_names=("spambase",),
        openml_data_ids=(44,),
    ),
    "pima_diabetes": DatasetSpec(
        app_id="pima_diabetes",
        display_name="Pima Indians Diabetes",
        source="openml",
        openml_names=("diabetes", "pima-indians-diabetes"),
        openml_data_ids=(37,),
    ),
    "ionosphere": DatasetSpec(
        app_id="ionosphere",
        display_name="Ionosphere",
        source="openml",
        openml_names=("ionosphere",),
        openml_data_ids=(59,),
    ),
    "blood_transfusion": DatasetSpec(
        app_id="blood_transfusion",
        display_name="Blood Transfusion Service Center",
        source="openml",
        openml_names=("blood-transfusion-service-center", "blood-transfusion"),
        openml_data_ids=(1464,),
    ),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def datasets_dir() -> Path:
    return project_root() / "datasets"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def backup_csv(path: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / f"{path.stem}_{timestamp}{path.suffix}")


def fetch_openml_frame(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    errors: list[str] = []

    for data_id in spec.openml_data_ids:
        try:
            log(f"    fetching OpenML data_id={data_id}...")
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            return bunch.data, bunch.target
        except Exception as exc:
            errors.append(f"data_id={data_id}: {exc}")

    for name in spec.openml_names:
        try:
            log(f"    fetching OpenML name={name!r}...")
            bunch = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
            return bunch.data, bunch.target
        except Exception as exc:
            errors.append(f"name={name}: {exc}")

    joined = "\n  ".join(errors)
    raise RuntimeError(f"Could not fetch {spec.app_id} from OpenML:\n  {joined}")


def load_raw_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.sklearn_loader is not None:
        return spec.sklearn_loader()
    if spec.source == "openml":
        return fetch_openml_frame(spec)
    raise ValueError(f"Unsupported dataset source for {spec.app_id}: {spec.source}")


def normalize_column_name(name: object) -> str:
    return str(name).replace("_", " ").strip().title()


def encode_binary_target(y_raw: pd.Series) -> tuple[np.ndarray, tuple[str, str]]:
    y = pd.Series(y_raw).dropna()
    if y.nunique(dropna=True) != 2:
        raise ValueError(f"Expected a binary target, found {y.nunique(dropna=True)} classes.")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    labels = tuple(str(cls) for cls in encoder.classes_)
    return y_encoded.astype(int), labels  # type: ignore[return-value]


def candidate_features(X: pd.DataFrame) -> list[dict]:
    candidates: list[dict] = []
    for col in X.columns:
        series = X[col]
        non_missing = series.dropna()
        if non_missing.empty or non_missing.nunique(dropna=True) < 2:
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_missing = numeric.dropna()
        if len(numeric_non_missing) >= max(10, int(0.8 * len(non_missing))):
            values = numeric.fillna(float(numeric_non_missing.median())).astype(float)
            candidates.append(
                {
                    "source_column": col,
                    "display_name": normalize_column_name(col),
                    "kind": "numeric",
                    "values": values.to_numpy(dtype=float),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "options": None,
                }
            )
            continue

        unique_values = sorted(str(v) for v in non_missing.unique())
        if len(unique_values) == 2:
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            encoded = series.astype(str).map(mapping).fillna(0).astype(float)
            candidates.append(
                {
                    "source_column": col,
                    "display_name": normalize_column_name(col),
                    "kind": "binary_categorical",
                    "values": encoded.to_numpy(dtype=float),
                    "min": np.nan,
                    "max": np.nan,
                    "options": unique_values,
                }
            )

    return candidates


def select_six_features(X: pd.DataFrame, y: np.ndarray) -> list[dict]:
    candidates = candidate_features(X)
    if len(candidates) < N_FEATURES:
        raise ValueError(f"Need at least {N_FEATURES} usable features, found {len(candidates)}.")

    matrix = np.column_stack([c["values"] for c in candidates])
    discrete = np.array([c["kind"] == "binary_categorical" for c in candidates], dtype=bool)
    scores = mutual_info_classif(
        matrix,
        y,
        discrete_features=discrete,
        random_state=RANDOM_STATE,
    )
    order = np.argsort(scores)[::-1][:N_FEATURES]
    selected = [candidates[int(i)].copy() for i in order]
    for new_idx, feature in enumerate(selected):
        feature["feature_index"] = new_idx
        feature["score"] = float(scores[int(order[new_idx])])
    return selected


def stratified_limit(X: pd.DataFrame, y: np.ndarray, max_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X.reset_index(drop=True), y
    _X_rest, X_sample, _y_rest, y_sample = train_test_split(
        X,
        y,
        test_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_sample.reset_index(drop=True), y_sample.astype(int)


def prepare_dataset(spec: DatasetSpec) -> PreparedDataset:
    log("  loading raw dataset...")
    X_raw_full, y_raw_full = load_raw_dataset(spec)
    frame = X_raw_full.copy()
    frame["__target__"] = y_raw_full
    frame = frame.dropna(subset=["__target__"]).reset_index(drop=True)
    y_encoded, target_labels = encode_binary_target(frame["__target__"])
    X_full = frame.drop(columns=["__target__"])

    log(f"  limiting rows to at most {MAX_ROWS_PER_DATASET}...")
    X_limited, y_limited = stratified_limit(X_full, y_encoded, MAX_ROWS_PER_DATASET)
    log("  selecting 6 usable attributes...")
    selected = select_six_features(X_limited, y_limited)
    selected_names = ", ".join(str(feature["display_name"]) for feature in selected)
    log(f"  selected features: {selected_names}")

    raw_columns: dict[str, np.ndarray] = {}
    model_columns: list[np.ndarray] = []
    feature_metadata: list[dict] = []

    for feature in selected:
        idx = int(feature["feature_index"])
        values = np.asarray(feature["values"], dtype=float)
        if len(values) != len(X_limited):
            source = X_limited[feature["source_column"]]
            if feature["kind"] == "numeric":
                numeric = pd.to_numeric(source, errors="coerce")
                values = numeric.fillna(float(numeric.median())).to_numpy(dtype=float)
            else:
                options = list(feature["options"])
                mapping = {value: i for i, value in enumerate(options)}
                values = source.astype(str).map(mapping).fillna(0).to_numpy(dtype=float)

        raw_columns[f"v{idx}"] = values
        if feature["kind"] == "numeric":
            lo = float(np.min(values))
            hi = float(np.max(values))
            scaled = np.zeros_like(values, dtype=float) if hi == lo else (values - lo) / (hi - lo)
            feature_metadata.append(
                {
                    "feature_index": idx,
                    "name": feature["display_name"],
                    "kind": "numeric",
                    "min": lo,
                    "max": hi,
                    "options": None,
                }
            )
        else:
            scaled = values.astype(float)
            feature_metadata.append(
                {
                    "feature_index": idx,
                    "name": feature["display_name"],
                    "kind": "binary_categorical",
                    "min": np.nan,
                    "max": np.nan,
                    "options": list(feature["options"]),
                }
            )
        model_columns.append(scaled)

    X_out = pd.DataFrame(raw_columns)
    X_model = np.column_stack(model_columns).astype(float)
    return PreparedDataset(
        app_id=spec.app_id,
        display_name=spec.display_name,
        X_raw=X_out,
        X_model=X_model,
        y=y_limited.astype(int),
        feature_metadata=feature_metadata,
        target_labels=target_labels,
    )


def train_models(prepared: PreparedDataset) -> dict[str, np.ndarray]:
    log("  splitting training data for AI models...")
    X_train, _X_test, y_train, _y_test = train_test_split(
        prepared.X_model,
        prepared.y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=prepared.y,
    )

    models = {
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-4,
                early_stopping=False,
                learning_rate_init=0.001,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
        "xgboost": XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        started = time.time()
        log(f"  training {model_name}...")
        model.fit(X_train, y_train)
        predictions[model_name] = np.asarray(model.predict(prepared.X_model), dtype=int)
        log(f"  finished {model_name} in {time.time() - started:.1f}s")
    return predictions


def fit_logistic_row(
    prepared: PreparedDataset,
    model_name: str,
    model_pred: np.ndarray,
    variant: str,
) -> dict:
    if len(np.unique(model_pred)) < 2:
        raise ValueError(f"{prepared.app_id}/{model_name} predictions have one class; cannot fit LR surrogate.")

    if variant == "dense":
        selected = tuple(range(N_FEATURES))
        C = np.nan
    elif variant == "sparse":
        best_score = -1.0
        best_combo = tuple(range(3))
        for combo in combinations(range(N_FEATURES), 3):
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
            lr.fit(prepared.X_model[:, combo], model_pred)
            score = accuracy_score(model_pred, lr.predict(prepared.X_model[:, combo]))
            if score > best_score:
                best_score = float(score)
                best_combo = combo
        selected = best_combo
        C = 1.0
    else:
        raise ValueError(f"Unknown LR variant: {variant}")

    lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
    lr.fit(prepared.X_model[:, selected], model_pred)
    pred = lr.predict(prepared.X_model[:, selected])
    coefs = np.zeros(N_FEATURES, dtype=float)
    coefs[list(selected)] = lr.coef_[0]

    row = {
        "appId": prepared.app_id,
        "model": model_name,
        "variant": variant,
        "fidelity": float(accuracy_score(model_pred, pred)),
        "intercept": float(lr.intercept_[0]),
        "C": C,
        "nnz": int(np.count_nonzero(np.abs(coefs) > 1e-10)),
        "k": 3 if variant == "sparse" else np.nan,
        "kept_groups": json.dumps([f"a{i}" for i in selected]) if variant == "sparse" else np.nan,
    }
    for i in range(N_FEATURES):
        row[f"coef_a{i}"] = float(coefs[i])
    return row


def tree_to_structure(tree: DecisionTreeClassifier) -> list[dict]:
    tree_ = tree.tree_
    structures: list[dict] = []
    for node_id in range(tree_.node_count):
        value = tree_.value[node_id][0].astype(float)
        total = float(value.sum())
        probs = (value / total).tolist() if total > 0 else [0.5, 0.5]
        is_leaf = tree_.feature[node_id] == _tree.TREE_UNDEFINED
        if is_leaf:
            feature = None
            threshold = None
            left = None
            right = None
        else:
            feature = f"a{int(tree_.feature[node_id])}"
            threshold = float(tree_.threshold[node_id])
            left = int(tree_.children_left[node_id])
            right = int(tree_.children_right[node_id])

        structures.append(
            {
                "node": int(node_id),
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": probs,
                "is_leaf": bool(is_leaf),
            }
        )
    return structures


def fit_tree_row(prepared: PreparedDataset, model_name: str, model_pred: np.ndarray, depth: int) -> dict:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(prepared.X_raw.to_numpy(dtype=float), model_pred)
    pred = tree.predict(prepared.X_raw.to_numpy(dtype=float))
    return {
        "appId": prepared.app_id,
        "model": model_name,
        "depth": int(depth),
        "fidelity": float(accuracy_score(model_pred, pred)),
        "tree_structure": json.dumps(tree_to_structure(tree)),
        "class_labels": json.dumps([0, 1]),
    }


def make_metadata_row(existing_columns: list[str], prepared: PreparedDataset) -> dict:
    row = {col: np.nan for col in existing_columns}
    row["appId"] = prepared.app_id
    row["y"] = prepared.display_name
    row["y0"] = prepared.target_labels[0]
    row["y1"] = prepared.target_labels[1]

    for feature in prepared.feature_metadata:
        idx = int(feature["feature_index"])
        row[f"a{idx}"] = feature["name"]
        if feature["kind"] == "numeric":
            row[f"v{idx}_min"] = float(feature["min"])
            row[f"v{idx}_max"] = float(feature["max"])
        else:
            options = list(feature["options"])
            row[f"v{idx}_options"] = float(len(options))
            for opt_idx, opt in enumerate(options):
                row[f"v{idx}_{opt_idx}"] = opt
    return row


def make_values_rows(existing_columns: list[str], prepared: PreparedDataset) -> pd.DataFrame:
    rows = []
    for instance_id, (_, feature_row) in enumerate(prepared.X_raw.iterrows()):
        row = {col: np.nan for col in existing_columns}
        row["appId"] = prepared.app_id
        row["instanceId"] = int(instance_id)
        for i in range(N_FEATURES):
            row[f"v{i}"] = float(feature_row[f"v{i}"])
        row["y"] = int(prepared.y[instance_id])
        rows.append(row)
    return pd.DataFrame(rows, columns=existing_columns)


def make_prediction_rows(existing_columns: list[str], prepared: PreparedDataset, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_name, pred in predictions.items():
        for instance_id, value in enumerate(pred):
            row = {col: np.nan for col in existing_columns}
            row["appId"] = prepared.app_id
            row["modelName"] = model_name
            row["instanceId"] = int(instance_id)
            row["pred"] = int(value)
            row["i_max"] = 0
            rows.append(row)
    return pd.DataFrame(rows, columns=existing_columns)


def align_columns(existing: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = list(existing.columns)
    for col in new_rows.columns:
        if col not in all_columns:
            all_columns.append(col)
    return existing.reindex(columns=all_columns), new_rows.reindex(columns=all_columns)


def replace_append(existing: pd.DataFrame, new_rows: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    existing_aligned, new_aligned = align_columns(existing, new_rows)
    if new_aligned.empty:
        return existing_aligned
    key_frame = new_aligned[key_cols].drop_duplicates()
    merged = existing_aligned.merge(key_frame.assign(__replace__=True), on=key_cols, how="left")
    kept = merged[merged["__replace__"].isna()].drop(columns=["__replace__"])
    return pd.concat([kept, new_aligned], ignore_index=True).reindex(columns=existing_aligned.columns)


def build_rows(prepared: PreparedDataset, csvs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    predictions = train_models(prepared)

    log("  building metadata/value/prediction rows...")
    metadata_rows = pd.DataFrame([make_metadata_row(list(csvs["metadata"].columns), prepared)])
    values_rows = make_values_rows(list(csvs["values"].columns), prepared)
    prediction_rows = make_prediction_rows(list(csvs["none"].columns), prepared, predictions)

    lr_rows = []
    tree_rows = []
    for model_name, model_pred in predictions.items():
        log(f"  fitting surrogate LR rows for {model_name}...")
        lr_rows.append(fit_logistic_row(prepared, model_name, model_pred, "dense"))
        lr_rows.append(fit_logistic_row(prepared, model_name, model_pred, "sparse"))
        log(f"  fitting surrogate tree rows for {model_name}...")
        tree_rows.append(fit_tree_row(prepared, model_name, model_pred, 2))
        tree_rows.append(fit_tree_row(prepared, model_name, model_pred, 3))

    return {
        "metadata": metadata_rows,
        "values": values_rows,
        "none": prediction_rows,
        "logistic_regression": pd.DataFrame(lr_rows),
        "decision_tree": pd.DataFrame(tree_rows),
    }


def load_project_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "metadata": read_csv(data_dir / "metadata.csv"),
        "values": read_csv(data_dir / "values.csv"),
        "none": read_csv(data_dir / "none.csv"),
        "logistic_regression": read_csv(data_dir / "logistic_regression.csv"),
        "decision_tree": read_csv(data_dir / "decision_tree.csv"),
    }


def write_project_csvs(data_dir: Path, csvs: dict[str, pd.DataFrame], *, backup: bool) -> None:
    path_by_name = {
        "metadata": data_dir / "metadata.csv",
        "values": data_dir / "values.csv",
        "none": data_dir / "none.csv",
        "logistic_regression": data_dir / "logistic_regression.csv",
        "decision_tree": data_dir / "decision_tree.csv",
    }
    for name, path in path_by_name.items():
        if backup:
            backup_csv(path)
        csvs[name].to_csv(path, index=False)


def append_generated_rows(csvs: dict[str, pd.DataFrame], generated: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "metadata": replace_append(csvs["metadata"], generated["metadata"], ["appId"]),
        "values": replace_append(csvs["values"], generated["values"], ["appId"]),
        "none": replace_append(csvs["none"], generated["none"], ["appId", "modelName"]),
        "logistic_regression": replace_append(
            csvs["logistic_regression"],
            generated["logistic_regression"],
            ["appId", "model", "variant"],
        ),
        "decision_tree": replace_append(
            csvs["decision_tree"],
            generated["decision_tree"],
            ["appId", "model", "depth"],
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append generated datasets to the project CSV files.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_SPECS),
        choices=sorted(DATASET_SPECS),
        help="Dataset appIds to generate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build rows and print counts without writing CSVs.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create timestamped CSV backups before writing.")
    parser.add_argument("--skip-failures", action="store_true", help="Skip datasets that fail to fetch or fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = datasets_dir()
    log(f"Loading project CSVs from {data_dir}...")
    csvs = load_project_csvs(data_dir)
    generated_totals = {name: [] for name in csvs}

    for app_id in args.datasets:
        spec = DATASET_SPECS[app_id]
        started = time.time()
        log(f"\nGenerating {spec.app_id} ({spec.display_name})...")
        try:
            prepared = prepare_dataset(spec)
            generated = build_rows(prepared, csvs)
        except Exception:
            if args.skip_failures:
                log(f"  skipped {spec.app_id}")
                continue
            raise

        for name, frame in generated.items():
            generated_totals[name].append(frame)
        log(
            f"  rows: values={len(generated['values'])}, "
            f"none={len(generated['none'])}, "
            f"lr={len(generated['logistic_regression'])}, "
            f"dt={len(generated['decision_tree'])}"
        )
        log(f"  completed {spec.app_id} in {time.time() - started:.1f}s")

    generated_csvs = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=csvs[name].columns)
        for name, frames in generated_totals.items()
    }
    updated = append_generated_rows(csvs, generated_csvs)

    log("\nGenerated row counts:")
    for name, frame in generated_csvs.items():
        log(f"  {name}.csv: +{len(frame)}")

    log("\nUpdated CSV shapes:")
    for name, frame in updated.items():
        log(f"  {name}.csv: {frame.shape}")

    if args.dry_run:
        log("\nDry run only; no files written.")
        return

    write_project_csvs(data_dir, updated, backup=not args.no_backup)
    log(f"\nWrote updated CSVs to {data_dir}")


if __name__ == "__main__":
    main()

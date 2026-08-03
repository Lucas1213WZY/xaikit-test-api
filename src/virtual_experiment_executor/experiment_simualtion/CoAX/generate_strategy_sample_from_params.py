import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parent_dir = os.getcwd()
sys.path.insert(0, parent_dir)

from data_loader import AIDatasetLoader
from experiment_runner import StrategyComparisonRunner
from human_models import AttributionSum, ImportanceCategorization, SalientFeatures, SensitiveFeatures
from ui import UI


DATA_DIR = Path(parent_dir) / "data" / "datasets" / "standard set"
PARAMS_PATH = Path("results/pop_em_subset/refit_all_assignments_detlocal-Jan-10.csv")
USER_STUDY_PATH = Path("data/user study results/3-datasets-jan-09-2026-trials.csv")


forest_blocks = [
    {
        "train": [24, 25, 154, 168, 183, 195, 215, 266, 292, 295],
        "testWithXAI": [21, 61, 102, 110, 130, 137, 151, 152, 179, 217, 223, 234, 239, 247, 270, 273, 278, 290],
        "testWithoutXAI": [8, 17, 22, 32, 53, 73, 81, 86, 95, 118, 122, 145, 172, 219, 220, 256, 260, 291],
    },
    {
        "train": [50, 78, 155, 163, 203, 206, 222, 225, 257, 298],
        "testWithXAI": [0, 44, 48, 65, 101, 135, 136, 139, 167, 175, 201, 207, 233, 236, 245, 246, 287, 288],
        "testWithoutXAI": [2, 9, 20, 41, 42, 70, 82, 89, 91, 94, 107, 109, 149, 177, 194, 205, 226, 274],
    },
]

wine_blocks = [
    {
        "train": [8, 25, 32, 43, 51, 66, 73, 81, 82, 121],
        "testWithXAI": [0, 4, 6, 22, 27, 36, 41, 42, 46, 62, 64, 65, 80, 86, 98, 101, 111, 117],
        "testWithoutXAI": [3, 13, 31, 44, 45, 52, 59, 67, 74, 75, 76, 78, 90, 91, 97, 105, 106, 110],
    },
    {
        "train": [7, 10, 24, 33, 40, 54, 87, 89, 114, 120],
        "testWithXAI": [20, 29, 34, 38, 39, 47, 56, 60, 71, 84, 88, 96, 100, 102, 103, 107, 112, 118],
        "testWithoutXAI": [1, 5, 14, 19, 21, 23, 26, 35, 55, 61, 68, 69, 77, 83, 108, 113, 116, 119],
    },
]

adult_blocks = [
    {
        "train": [14, 17, 119, 141, 168, 169, 213, 215, 260, 289],
        "testWithXAI": [2, 33, 35, 75, 76, 84, 95, 117, 125, 135, 158, 172, 190, 194, 210, 235, 246, 261],
        "testWithoutXAI": [5, 6, 18, 58, 81, 132, 145, 156, 161, 165, 171, 179, 221, 275, 276, 277, 294, 296],
    },
    {
        "train": [45, 47, 139, 187, 198, 212, 255, 263, 293, 295],
        "testWithXAI": [3, 19, 23, 26, 29, 66, 78, 97, 101, 114, 121, 150, 184, 207, 232, 267, 281, 282],
        "testWithoutXAI": [4, 20, 49, 50, 89, 94, 103, 111, 120, 224, 240, 242, 243, 268, 269, 290, 291, 298],
    },
]


DATASETS = {
    "forest_cover": {"name": "Forest Cover", "dataId": "forest_cover", "model": "xgboost", "expMethod": "shap", "blocks": forest_blocks},
    "wine_quality": {"name": "Wine Quality", "dataId": "wine_quality", "model": "mlp", "expMethod": "lime", "blocks": wine_blocks},
    "adult": {"name": "Adult Income", "dataId": "adult", "model": "xgboost", "expMethod": "lime", "blocks": adult_blocks},
}

STRATEGIES = {
    "Attribution Sum": AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
}

STRATS_BY_XAI = {
    "None": ["Sensitive-features categorization"],
    "Importance": [
        "Sensitive-features categorization",
        "Salient-features categorization",
        "Importance categorization",
        "Attribution Sum",
    ],
    "Attribution": ["Sensitive-features categorization", "Attribution Sum"],
}

TESTED_BY_XAI = {
    "None": ["w/o XAI"],
    "Importance": ["w/ XAI", "w/o XAI"],
    "Attribution": ["w/ XAI", "w/o XAI"],
}

PARAM_BOUNDS = {
    "k": (1, 5),
    "sensitivity": (1.0, 20.0),
    "retrieval_threshold": (-2.3, -1.3),
    "scaling_factor": (1.0, 7.0),
}


def canonical_xai_type(value):
    key = str(value).strip().lower()
    aliases = {"none": "None", "no": "None", "importance": "Importance", "attribution": "Attribution"}
    if key not in aliases:
        raise ValueError(f"Unknown XAIType '{value}'. Use None, Importance, or Attribution.")
    return aliases[key]


def canonical_condition(value):
    key = str(value).strip().lower()
    if key in {"w/ xai", "xai", "with xai", "with"}:
        return "w/ XAI"
    if key in {"w/o xai", "no xai", "without xai", "without"}:
        return "w/o XAI"
    raise ValueError(f"Unknown tested condition '{value}'. Use 'w/ XAI' or 'w/o XAI'.")


def available_strategies(xai_type, tested_condition):
    strategies = list(STRATS_BY_XAI[xai_type])
    if xai_type == "Importance" and tested_condition == "w/o XAI":
        strategies = [strategy for strategy in strategies if strategy != "Importance categorization"]
    if xai_type == "Attribution" and tested_condition == "w/ XAI":
        strategies = [strategy for strategy in strategies if strategy != "Sensitive-features categorization"]
    return strategies


def make_mask(df, **wanted):
    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        if col in df.columns:
            mask &= df[col] == val
    return mask


def make_ai_loader(ds_cfg, xai_type):
    df_values = pd.read_csv(DATA_DIR / "values.csv")
    df_metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    expl_name = "attribution.csv" if xai_type == "Attribution" else "importance.csv"
    df_expl = pd.read_csv(DATA_DIR / expl_name)

    return (
        AIDatasetLoader(
            feature_values_df=df_values,
            metadata_df=df_metadata,
            explanation_values_df=df_expl,
            explanation_columns=["a0_i", "a1_i", "a2_i", "a3_i", "a4_i"] if xai_type != "None" else [],
        )
        .filter_loader(
            lambda df, data_id=ds_cfg["dataId"], mdl=ds_cfg["model"], exp=ds_cfg["expMethod"]: make_mask(
                df, dataId=data_id, modelName=mdl, expMethod=exp
            )
        )
    )


def class_labels(ds_cfg):
    df_metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    row = df_metadata[df_metadata["dataId"] == ds_cfg["dataId"]]
    if row.empty:
        return {0: "0", 1: "1"}
    row = row.iloc[0]
    return {0: row["y0"], 1: row["y1"]}


def label_choice(value, labels):
    if value is None or pd.isna(value):
        return value
    try:
        return labels.get(int(value), str(value))
    except (TypeError, ValueError):
        return value


def clamp(value, param_name):
    low, high = PARAM_BOUNDS[param_name]
    return min(max(value, low), high)


def target_instance_id(ds_cfg, tested_condition, instance_number, block_index):
    block = ds_cfg["blocks"][block_index]
    key = "testWithXAI" if tested_condition == "w/ XAI" else "testWithoutXAI"
    ids = block[key]
    if instance_number < 0 or instance_number >= len(ids):
        raise IndexError(f"instance_number must be 0..{len(ids) - 1} for {ds_cfg['dataId']} {key}.")
    return ids[instance_number]


def target_instance_location(ds_cfg, tested_condition, instance_id, block_index=None):
    key = "testWithXAI" if tested_condition == "w/ XAI" else "testWithoutXAI"
    blocks = ds_cfg["blocks"]
    block_indexes = [block_index] if block_index is not None else range(len(blocks))
    for idx in block_indexes:
        ids = blocks[idx][key]
        if instance_id in ids:
            return idx, ids.index(instance_id)
    label = ds_cfg.get("dataId", ds_cfg.get("name", "dataset"))
    raise ValueError(f"Instance {instance_id} is not a {tested_condition} test instance for {label}.")


def make_one_block_session(ds_cfg, tested_condition, instance_id, block_index, train_with_explanation=True):
    block = ds_cfg["blocks"][block_index]
    session = []

    for train_id in random.sample(block["train"], len(block["train"])):
        session.append(
            {
                "instance_id": train_id,
                "is_training": True,
                "with_explanation": bool(train_with_explanation),
            }
        )

    session.append(
        {
            "instance_id": instance_id,
            "is_training": False,
            "with_explanation": tested_condition == "w/ XAI",
        }
    )
    return session


def strategy_config(args, strategy_name, xai_type):
    config = {
        "k": int(round(clamp(int(args.k), "k"))),
        "decay_param": float(args.decay_param),
        "retrieval_threshold": float(clamp(float(args.retrieval_threshold), "retrieval_threshold")),
    }
    if strategy_name == "Attribution Sum":
        config["scaling_factor"] = float(clamp(float(args.scaling_factor), "scaling_factor"))
        config["explanation_type"] = xai_type.lower()
    else:
        config["sensitivity"] = float(clamp(float(args.sensitivity), "sensitivity"))
    return config


def normalize_probs(probs):
    clean = {int(k): float(v) for k, v in probs.items() if pd.notna(v)}
    total = sum(clean.values())
    if total <= 0:
        return {0: 0.5, 1: 0.5}
    return {k: v / total for k, v in clean.items()}


def argmax_choice(probs):
    if not probs:
        return np.nan
    return max(sorted(probs.keys()), key=lambda label: probs[label])


def run_strategy(args, ds_cfg, ai_loader, xai_type, tested_condition, strategy_name, instance_id):
    distributions = []
    ai_prediction = None

    for session_idx in range(args.n_sessions):
        random.seed(args.seed + session_idx)
        np.random.seed(args.seed + session_idx)

        config = strategy_config(args, strategy_name, xai_type)
        strategy = STRATEGIES[strategy_name](**config)
        runner = StrategyComparisonRunner(strategy, ai_loader, UI())
        session = make_one_block_session(
            ds_cfg,
            tested_condition,
            instance_id,
            args.block_index,
            train_with_explanation=args.train_with_explanation,
        )

        logs = runner.generalized_run_experiment(session)
        target_logs = [
            lg
            for lg in logs
            if lg.get("Step") == "infer"
            and lg.get("instance_id") == instance_id
            and ("w/ XAI" if lg.get("explanation") is not None else "w/o XAI") == tested_condition
        ]
        if not target_logs:
            raise RuntimeError(f"No matching target inference log for {strategy_name}.")

        log = target_logs[-1]
        distributions.append(normalize_probs(log["response"]))
        ai_prediction = log["ai_prediction"]

    labels = sorted({label for dist in distributions for label in dist.keys()} | {0, 1})
    avg_dist = {label: float(np.mean([dist.get(label, 0.0) for dist in distributions])) for label in labels}
    avg_dist = normalize_probs(avg_dist)

    selected_choice = int(argmax_choice(avg_dist))

    return {
        "Strategy": strategy_name,
        "CoAX Choice": selected_choice,
        "CoAX Confidence": float(avg_dist.get(selected_choice, 0.0)),
        "AI Prediction": int(ai_prediction) if pd.notna(ai_prediction) else ai_prediction,
    }


def closest_participants(df_params, args, ds_cfg, xai_type, tested_condition, strategy_name):
    sub = df_params[
        (df_params["dataId"] == ds_cfg["dataId"])
        & (df_params["XAIType"].astype(str).str.lower() == xai_type.lower())
        & (df_params["Tested w/ XAI"] == tested_condition)
        & (df_params["Strategy"] == strategy_name)
    ].copy()

    param_cols = ["k", "retrieval_threshold"]
    target = {
        "k": float(clamp(float(args.k), "k")),
        "retrieval_threshold": float(clamp(float(args.retrieval_threshold), "retrieval_threshold")),
    }
    if strategy_name == "Attribution Sum":
        param_cols.append("scaling_factor")
        target["scaling_factor"] = float(clamp(float(args.scaling_factor), "scaling_factor"))
    else:
        param_cols.append("sensitivity")
        target["sensitivity"] = float(clamp(float(args.sensitivity), "sensitivity"))

    sub = sub.dropna(subset=param_cols)
    if sub.empty:
        return pd.DataFrame()

    for col in param_cols:
        sub[col] = sub[col].astype(float).map(lambda value, name=col: clamp(value, name))

    for col in param_cols:
        std = sub[col].astype(float).std(ddof=0)
        scale = float(std) if std and not np.isnan(std) and std >= 1e-6 else 1.0
        sub[f"_d_{col}"] = (sub[col].astype(float) - target[col]) / scale

    dist_cols = [f"_d_{col}" for col in param_cols]
    sub["Parameter Distance"] = np.sqrt((sub[dist_cols] ** 2).sum(axis=1))
    keep_cols = [
        "Participant ID",
        "dataId",
        "Session",
        "Tested w/ XAI",
        "XAIType",
        "Strategy",
        "Assignment Prob",
        "NLL",
        "PAI",
        "MAI",
        "sensitivity",
        "k",
        "retrieval_threshold",
        "scaling_factor",
        "Parameter Distance",
    ]
    keep_cols = [col for col in keep_cols if col in sub.columns]
    return sub.nsmallest(args.closest_k, "Parameter Distance")[keep_cols]


def load_human_responses(path):
    df = pd.read_csv(path, keep_default_na=False)
    df["Step"] = df["Step"].astype(str)
    return df


def response_distribution(df_user, nearest, args, ds_cfg, xai_type, tested_condition, strategy_name, instance_id):
    if nearest.empty:
        return {
            "Human Response": np.nan,
            "Human Confidence": np.nan,
            "Human n": 0,
        }, pd.DataFrame()

    participant_ids = nearest["Participant ID"].dropna().astype(str).unique().tolist()
    sub = df_user[
        (df_user["Participant ID"].astype(str).isin(participant_ids))
        & (df_user["Instance Index"] == instance_id)
        & (df_user["dataId"] == ds_cfg["dataId"])
        & (df_user["expMethod"] == ds_cfg["expMethod"])
        & (df_user["modelName"] == ds_cfg["model"])
        & (df_user["Session"] == args.session)
        & (df_user["XAIType"].astype(str).str.lower() == xai_type.lower())
        & (df_user["Tested w/ XAI"] == tested_condition)
        & (df_user["trialType"].astype(str).str.lower() == "test")
        & (df_user["Step"].astype(str).str.lower() == "infer")
    ].copy()

    sub["Strategy"] = strategy_name
    sub = sub[sub["Response"].notna() & (sub["Response"].astype(str).str.strip() != "")]
    matched_ids = sub["Participant ID"].dropna().astype(str).unique().tolist()
    missing_ids = [pid for pid in participant_ids if pid not in matched_ids]

    if sub.empty:
        return {
            "Human Response": np.nan,
            "Human Confidence": np.nan,
            "Human n": 0,
        }, sub

    responses = sub["Response"].astype(float).astype(int)
    counts = responses.value_counts().sort_index()
    total = int(counts.sum())
    probs = {int(label): float(count / total) for label, count in counts.items()}

    selected_response = int(argmax_choice(probs))

    return {
        "Human Response": selected_response,
        "Human Confidence": float(probs.get(selected_response, 0.0)),
        "Human n": total,
    }, sub


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a targeted CoAX simulation from cognitive parameters and report argmax strategy-level predictions."
    )
    parser.add_argument("--dataset", default="forest_cover", choices=sorted(DATASETS.keys()))
    parser.add_argument("--xai-type", default="Importance")
    parser.add_argument("--reasoning-strategy", default=None, choices=sorted(STRATEGIES.keys()))
    parser.add_argument("--tested", default="w/ XAI", help="'w/ XAI' or 'w/o XAI'")
    parser.add_argument("--instance-number", type=int, required=True)
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--retrieval-threshold", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, default=20.0)
    parser.add_argument("--scaling-factor", type=float, default=1.0)
    parser.add_argument("--decay-param", type=float, default=0.5)
    parser.add_argument("--n-sessions", type=int, default=2)
    parser.add_argument("--closest-k", type=int, default=7)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-with-explanation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-study-path", default=str(USER_STUDY_PATH))
    parser.add_argument("--output-prefix", default="query")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def main():
    payload = build_payload(parse_args())
    print(json.dumps(json_safe(payload), indent=2))


def build_payload(args):
    xai_type = canonical_xai_type(args.xai_type)
    tested_condition = canonical_condition(args.tested)
    ds_cfg = DATASETS[args.dataset]
    available_conditions = TESTED_BY_XAI[xai_type]
    if tested_condition not in available_conditions:
        raise ValueError(
            f"Tested condition '{tested_condition}' is not available for XAIType '{xai_type}'. "
            f"Use one of: {', '.join(available_conditions)}."
        )
    available_strategy_names = available_strategies(xai_type, tested_condition)
    strategy_name = getattr(args, "reasoning_strategy", None) or available_strategy_names[0]
    if strategy_name not in available_strategy_names:
        raise ValueError(
            f"Strategy '{strategy_name}' is not available for XAIType '{xai_type}' "
            f"under '{tested_condition}'. Use one of: {', '.join(available_strategy_names)}."
        )
    requested_instance_id = getattr(args, "instance_id", None)
    if requested_instance_id is None:
        instance_id = target_instance_id(ds_cfg, tested_condition, args.instance_number, args.block_index)
    else:
        try:
            block_index, instance_number = target_instance_location(
                ds_cfg,
                tested_condition,
                int(requested_instance_id),
                getattr(args, "block_index", None),
            )
        except ValueError:
            block_index = int(getattr(args, "block_index", 0))
            instance_number = int(getattr(args, "instance_number", 0))
        args.block_index = block_index
        args.instance_number = instance_number
        instance_id = int(requested_instance_id)

    ai_loader = make_ai_loader(ds_cfg, xai_type)
    labels = class_labels(ds_cfg)
    df_params = pd.read_csv(PARAMS_PATH, keep_default_na=False)
    df_user = load_human_responses(args.user_study_path)

    coax_row = run_strategy(args, ds_cfg, ai_loader, xai_type, tested_condition, strategy_name, instance_id)

    nearest = closest_participants(df_params, args, ds_cfg, xai_type, tested_condition, strategy_name)
    human_summary, _human_matches = response_distribution(
        df_user, nearest, args, ds_cfg, xai_type, tested_condition, strategy_name, instance_id
    )

    row = {**coax_row, **human_summary}
    row["CoAX Choice"] = label_choice(row["CoAX Choice"], labels)
    row["Human Response"] = label_choice(row["Human Response"], labels)
    row["AI Prediction"] = label_choice(row["AI Prediction"], labels)

    payload = {
        "dataset": ds_cfg["dataId"],
        "xai_type": xai_type,
        "tested_condition": tested_condition,
        "instance_number": args.instance_number,
        "instance_id": instance_id,
        "strategy_predictions": [row],
    }
    return json_safe(payload)


if __name__ == "__main__":
    main()

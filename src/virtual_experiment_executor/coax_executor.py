"""Virtual experiment execution API for CoAX-style simulated trial rows.

This module replaces the old `code_for_papers/old/coax/generate_data_sample_gaussian.py`
workflow with:
  - src.data_loaders.UnifiedDataLoader for instance/prediction/explanation access
  - src.cognitive_models for strategy instantiation and inference
  - fitted participant cognitive parameters from assets/human_trials_and_cogntive_parameters

For each participant/appId/XAIType/condition group, the script simulates every
eligible strategy available in the parameter CSV and writes the trial rows. Use
`--select-best` if you later want to keep only the lowest simulated NLL strategy.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_loaders import UnifiedDataLoader
from src.cognitive_models import (
    ReasoningMode,
    StrategyConfig,
    StrategyRegistry,
    StrategyType,
    initialize_strategies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = REPO_ROOT / "assets"
DEFAULT_COAX_PARAMS_PATH = DEFAULT_ASSETS_ROOT / "human_trials_and_cogntive_parameters" / "CoAX_cog_param.csv"


DATASETS: Dict[str, Dict[str, Any]] = {
    "forest_cover": {
        "model": "xgboost",
        "exp_method": "shap",
        "blocks": [
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
        ],
    },
    "wine_quality": {
        "model": "mlp",
        "exp_method": "lime",
        "blocks": [
            {
                "train": [8, 25, 32, 43, 51, 66, 73, 81, 82],
                "testWithXAI": [0, 4, 6, 22, 27, 36, 41, 42, 46, 62, 64, 65, 80, 86, 98, 101, 111, 117],
                "testWithoutXAI": [3, 13, 31, 44, 45, 52, 59, 67, 74, 75, 76, 78, 90, 91, 97, 105, 106, 110],
            },
            {
                "train": [7, 10, 24, 33, 40, 54, 87, 89, 114],
                "testWithXAI": [20, 29, 34, 38, 39, 47, 56, 60, 71, 84, 88, 96, 100, 102, 103, 107, 112, 118],
                "testWithoutXAI": [1, 5, 14, 19, 21, 23, 26, 35, 55, 61, 68, 69, 77, 83, 108, 113, 116, 119],
            },
        ],
    },
    "adult": {
        "model": "xgboost",
        "exp_method": "lime",
        "blocks": [
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
        ],
    },
}

STRATEGY_NAME_MAP = {
    "Sensitive-features categorization": "sensitive_features",
    "Salient-features categorization": "salient_features",
    "Importance categorization": "importance_categorization",
    "Attribution Sum": "attribution_sum",
}

CSV_COLUMNS = [
    "Participant Id",
    "appId",
    "XAIType",
    "Tested w/ XAI",
    "Selected Strategy",
    "Selected Strategy NLL",
    "Strategy",
    "Strategy NLL",
    "Parameter Row Index",
    "Parameter NLL",
    "Parameter Session",
    "k",
    "decay_param",
    "sensitivity",
    "retrieval_threshold",
    "scaling_factor",
    "explanation_type",
    "Instance Index",
    "trialType",
    "predicted",
    "Prob correct",
    "Prob 0",
    "Prob 1",
    "Correct",
    "response_time",
]


@dataclass
class Trial:
    instance_id: int
    is_training: bool
    with_explanation: bool


@dataclass
class SimulationResult:
    nll: float
    rows: List[Dict[str, Any]]


@dataclass
class VirtualExperimentConfig:
    """Configuration for generating simulated CoAX virtual-experiment rows."""

    params_path: str = str(DEFAULT_COAX_PARAMS_PATH)
    assets_root: str = str(DEFAULT_ASSETS_ROOT)
    app_id: str = "all"
    xai_type: str = "Importance"
    tested_with_xai: str = "all"
    max_participants: Optional[int] = None
    seed: int = 7
    select_best: bool = False


def normalize_xai_type(value: Any) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text:
        return "None"
    return text[:1].upper() + text[1:].lower()


def make_session(blocks: Sequence[Dict[str, Sequence[int]]], rng: random.Random, train_with_explanation: bool) -> List[Trial]:
    session: List[Trial] = []
    for block in blocks:
        for key in ("train", "testWithXAI", "testWithoutXAI"):
            ids = list(block[key])
            rng.shuffle(ids)
            for instance_id in ids:
                if key == "train":
                    session.append(Trial(instance_id, is_training=True, with_explanation=train_with_explanation))
                elif key == "testWithXAI":
                    session.append(Trial(instance_id, is_training=False, with_explanation=True))
                else:
                    session.append(Trial(instance_id, is_training=False, with_explanation=False))
    return session


def strategy_config_from_row(row: pd.Series, registry_name: str) -> StrategyConfig:
    extra_params: Dict[str, Any] = {}

    if "k" in row.index and not pd.isna(row["k"]):
        extra_params["k"] = int(round(float(row["k"])))
    if "sensitivity" in row.index and not pd.isna(row["sensitivity"]):
        extra_params["sensitivity"] = float(row["sensitivity"])
    if "scaling_factor" in row.index and not pd.isna(row["scaling_factor"]):
        extra_params["scaling_factor"] = float(row["scaling_factor"])
    if "explanation_type" in row.index and not pd.isna(row["explanation_type"]):
        extra_params["explanation_type"] = str(row["explanation_type"])

    decay_param = float(row["decay_param"]) if "decay_param" in row.index and not pd.isna(row["decay_param"]) else 0.5
    retrieval_threshold = (
        float(row["retrieval_threshold"])
        if "retrieval_threshold" in row.index and not pd.isna(row["retrieval_threshold"])
        else -2.5
    )

    return StrategyConfig(
        strategy_name=registry_name,
        strategy_type=StrategyType.COAX_FORWARD,
        mode=ReasoningMode.READ,
        decay_param=decay_param,
        retrieval_threshold=retrieval_threshold,
        sensitivity=float(extra_params.get("sensitivity", 10.0)),
        extra_params=extra_params,
    )


def load_coax_loader(assets_root: str, app_id: str, xai_type: str) -> UnifiedDataLoader:
    explanation_type = "attribution" if xai_type == "Attribution" else "importance"
    loader = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root=assets_root,
        app_id=app_id,
        coax_explanation_type=explanation_type,
    )

    if xai_type == "None":
        none_file = Path(assets_root) / "data" / "coax" / "none.csv"
        pred_df = pd.read_csv(none_file)
    else:
        pred_df = loader.get_explanation_table(explanation_type)

    ds_cfg = DATASETS[app_id]
    filtered = pred_df[pred_df["appId"].eq(app_id)].copy()
    if "modelName" in filtered.columns:
        filtered = filtered[filtered["modelName"].eq(ds_cfg["model"])].copy()
    if xai_type != "None" and "expMethod" in filtered.columns:
        filtered = filtered[filtered["expMethod"].eq(ds_cfg["exp_method"])].copy()

    loader.data_source.ai_predictions_df = filtered
    loader.data_source.explanation_columns = [
        c for c in filtered.columns if c.startswith("a") and c.endswith("_i")
    ]
    return loader


def load_trial_inputs(loader: UnifiedDataLoader, instance_id: int) -> Tuple[np.ndarray, int, np.ndarray]:
    features, predictions = loader.get_instances([instance_id])
    explanations = loader.get_explanations([instance_id])

    feature_array = np.asarray(features[0], dtype=float)
    ai_prediction = int(predictions[0])
    explanation = explanations[0]
    if isinstance(explanation, dict):
        explanation_array = np.asarray(
            [explanation.get(col, 0.0) or 0.0 for col in loader.data_source.explanation_columns],
            dtype=float,
        )
    else:
        explanation_array = np.asarray(explanation, dtype=float)
    return feature_array, ai_prediction, explanation_array


def normalize_probs(probs: Dict[Any, float]) -> Dict[int, float]:
    cleaned = {int(k): max(0.0, float(v)) for k, v in probs.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {0: 0.5, 1: 0.5}
    return {0: cleaned.get(0, 0.0) / total, 1: cleaned.get(1, 0.0) / total}


def sample_choice(probs: Dict[int, float], rng: np.random.Generator) -> int:
    norm = normalize_probs(probs)
    return int(rng.choice([0, 1], p=[norm[0], norm[1]]))


def simulate_candidate(
    *,
    loader: UnifiedDataLoader,
    session: Sequence[Trial],
    param_row: pd.Series,
    participant_id: str,
    app_id: str,
    xai_type: str,
    tested_with_xai: str,
    rng: np.random.Generator,
) -> SimulationResult:
    public_strategy_name = str(param_row["Strategy"])
    registry_name = STRATEGY_NAME_MAP[public_strategy_name]
    strategy = StrategyRegistry.get(registry_name, strategy_config_from_row(param_row, registry_name))

    rows: List[Dict[str, Any]] = []
    losses: List[float] = []

    for trial in session:
        strategy.new_instance()
        features, ai_prediction, explanation = load_trial_inputs(loader, trial.instance_id)

        trial_tested = "w/ XAI" if trial.with_explanation else "w/o XAI"

        if (not trial.with_explanation) or trial.is_training:
            probs, response_time, _info = strategy.infer(
                features=features,
                explanation=None,
                ai_prediction=ai_prediction,
            )
            if trial_tested == tested_with_xai:
                probs = normalize_probs(probs)
                choice = sample_choice(probs, rng)
                prob_correct = max(probs.get(ai_prediction, 0.0), 1e-12)
                losses.append(-math.log(prob_correct))
                rows.append(
                    trial_row(
                        param_row=param_row,
                        participant_id=participant_id,
                        app_id=app_id,
                        xai_type=xai_type,
                        tested_with_xai=trial_tested,
                        instance_id=trial.instance_id,
                        trial_type="Train" if trial.is_training else "Test",
                        choice=choice,
                        ai_prediction=ai_prediction,
                        probs=probs,
                        response_time=response_time,
                    )
                )

        if trial.with_explanation:
            probs, response_time, _info = strategy.infer(
                features=features,
                explanation=explanation,
                ai_prediction=ai_prediction,
            )
            if trial_tested == tested_with_xai:
                probs = normalize_probs(probs)
                choice = sample_choice(probs, rng)
                prob_correct = max(probs.get(ai_prediction, 0.0), 1e-12)
                losses.append(-math.log(prob_correct))
                rows.append(
                    trial_row(
                        param_row=param_row,
                        participant_id=participant_id,
                        app_id=app_id,
                        xai_type=xai_type,
                        tested_with_xai=trial_tested,
                        instance_id=trial.instance_id,
                        trial_type="Train" if trial.is_training else "Test",
                        choice=choice,
                        ai_prediction=ai_prediction,
                        probs=probs,
                        response_time=response_time,
                    )
                )

        if trial.is_training:
            strategy.feedback(
                features=features,
                true_label=ai_prediction,
                explanation=explanation if trial.with_explanation else None,
            )

    if not losses:
        return SimulationResult(nll=float("inf"), rows=rows)
    return SimulationResult(nll=float(np.mean(losses)), rows=rows)


def trial_row(
    *,
    param_row: pd.Series,
    participant_id: str,
    app_id: str,
    xai_type: str,
    tested_with_xai: str,
    instance_id: int,
    trial_type: str,
    choice: int,
    ai_prediction: int,
    probs: Dict[int, float],
    response_time: float,
) -> Dict[str, Any]:
    return {
        "Participant Id": participant_id,
        "appId": app_id,
        "XAIType": "" if xai_type == "None" else xai_type,
        "Tested w/ XAI": tested_with_xai,
        "Strategy": param_row["Strategy"],
        "Strategy NLL": np.nan,
        "Parameter Row Index": param_row.name,
        "Parameter NLL": param_row.get("NLL", np.nan),
        "Parameter Session": param_row.get("Session", np.nan),
        "k": param_row.get("k", np.nan),
        "decay_param": param_row.get("decay_param", 0.5),
        "sensitivity": param_row.get("sensitivity", np.nan),
        "retrieval_threshold": param_row.get("retrieval_threshold", np.nan),
        "scaling_factor": param_row.get("scaling_factor", np.nan),
        "explanation_type": param_row.get("explanation_type", xai_type.lower()),
        "Instance Index": instance_id,
        "trialType": trial_type,
        "predicted": choice,
        "Prob correct": probs.get(ai_prediction, 0.0),
        "Prob 0": probs.get(0, np.nan),
        "Prob 1": probs.get(1, np.nan),
        "Correct": int(choice == ai_prediction),
        "response_time": response_time,
    }


def candidate_param_rows(
    df_params: pd.DataFrame,
    *,
    participant_id: str,
    app_id: str,
    xai_type: str,
    tested_with_xai: str,
) -> pd.DataFrame:
    sub = df_params[
        df_params["Participant Id"].astype(str).eq(str(participant_id))
        & df_params["appId"].astype(str).eq(app_id)
        & df_params["_xai_type_normalized"].eq(xai_type)
        & df_params["Tested w/ XAI"].astype(str).eq(tested_with_xai)
        & df_params["Strategy"].isin(STRATEGY_NAME_MAP)
    ].copy()

    if sub.empty:
        return sub

    # If a participant has duplicate rows for the same strategy/session, keep the
    # lowest fitted NLL row as the parameter representative for that strategy.
    sub["_nll_sort"] = pd.to_numeric(sub["NLL"], errors="coerce").fillna(float("inf"))
    sub = sub.sort_values("_nll_sort").drop_duplicates(["Strategy"], keep="first")
    return sub.drop(columns=["_nll_sort"])


def iter_participants(df_params: pd.DataFrame, app_id: str, xai_type: str, tested_with_xai: str) -> Iterable[str]:
    sub = df_params[
        df_params["appId"].astype(str).eq(app_id)
        & df_params["_xai_type_normalized"].eq(xai_type)
        & df_params["Tested w/ XAI"].astype(str).eq(tested_with_xai)
    ]
    return sub["Participant Id"].dropna().astype(str).drop_duplicates().tolist()


def simulate_virtual_experiment(config: VirtualExperimentConfig) -> pd.DataFrame:
    """Generate simulated trial rows as a DataFrame."""
    initialize_strategies()

    df_params = pd.read_csv(config.params_path)
    df_params["_xai_type_normalized"] = df_params["XAIType"].apply(normalize_xai_type)

    app_ids = sorted(DATASETS) if config.app_id == "all" else [config.app_id]
    xai_types = ["Importance", "Attribution", "None"] if config.xai_type == "all" else [config.xai_type]
    tested_conditions = ["w/ XAI", "w/o XAI"] if config.tested_with_xai == "all" else [config.tested_with_xai]

    all_rows: List[Dict[str, Any]] = []
    skipped = 0
    fake_participant_counter = 0

    for app_id in app_ids:
        for xai_type in xai_types:
            if xai_type == "None" and not df_params["_xai_type_normalized"].eq("None").any():
                continue

            loader = load_coax_loader(config.assets_root, app_id, xai_type)

            for tested_with_xai in tested_conditions:
                participant_ids = list(iter_participants(df_params, app_id, xai_type, tested_with_xai))
                if config.max_participants is not None:
                    participant_ids = participant_ids[: config.max_participants]

                for participant_index, participant_id in enumerate(participant_ids):
                    candidates = candidate_param_rows(
                        df_params,
                        participant_id=participant_id,
                        app_id=app_id,
                        xai_type=xai_type,
                        tested_with_xai=tested_with_xai,
                    )
                    if candidates.empty:
                        skipped += 1
                        continue

                    train_with_explanation = xai_type != "None"
                    session_rng = random.Random(config.seed + participant_index)
                    session = make_session(DATASETS[app_id]["blocks"], session_rng, train_with_explanation)

                    candidate_results: List[Tuple[pd.Series, SimulationResult]] = []
                    fake_participant_counter += 1
                    simulated_participant_id = fake_participant_counter
                    for _, param_row in candidates.iterrows():
                        sim_rng = np.random.default_rng(config.seed + participant_index)
                        result = simulate_candidate(
                            loader=loader,
                            session=session,
                            param_row=param_row,
                            participant_id=simulated_participant_id,
                            app_id=app_id,
                            xai_type=xai_type,
                            tested_with_xai=tested_with_xai,
                            rng=sim_rng,
                        )
                        candidate_results.append((param_row, result))

                    if not candidate_results:
                        skipped += 1
                        continue

                    if config.select_best:
                        candidate_results = [min(candidate_results, key=lambda item: item[1].nll)]

                    for param_row, result in candidate_results:
                        if config.select_best:
                            for row in result.rows:
                                row["Participant Id"] = simulated_participant_id
                        for row in result.rows:
                            row["Selected Strategy"] = param_row["Strategy"] if config.select_best else ""
                            row["Selected Strategy NLL"] = result.nll if config.select_best else np.nan
                            row["Strategy NLL"] = result.nll
                        all_rows.extend(result.rows)

    df_out = pd.DataFrame(all_rows)
    for col in CSV_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    return df_out[CSV_COLUMNS]


def save_virtual_experiment_results(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save simulated virtual-experiment rows to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_COAX_PARAMS_PATH))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--output", default="simulated_results/coax_api_simulated_trials.csv")
    parser.add_argument("--app-id", default="all", choices=["all", *sorted(DATASETS)])
    parser.add_argument("--xai-type", default="Importance", choices=["all", "Importance", "Attribution", "None"])
    parser.add_argument("--tested-with-xai", default="all", choices=["all", "w/ XAI", "w/o XAI"])
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="Keep only the lowest simulated NLL strategy per participant/app/XAI/condition.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = VirtualExperimentConfig(
        params_path=args.params,
        assets_root=args.assets_root,
        app_id=args.app_id,
        xai_type=args.xai_type,
        tested_with_xai=args.tested_with_xai,
        max_participants=args.max_participants,
        seed=args.seed,
        select_best=args.select_best,
    )
    df_out = simulate_virtual_experiment(config)
    output_path = save_virtual_experiment_results(df_out, args.output)
    print(f"Wrote {len(df_out)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()

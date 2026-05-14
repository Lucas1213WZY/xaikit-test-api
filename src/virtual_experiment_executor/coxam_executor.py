"""Virtual experiment execution API for CoXAM-style simulated trial rows.

Parallel to coax_executor.py but uses the CoXAM path:
  - Trained PPO meta-model selects strategy per trial (DT / LR Calc / LR Heur)
  - Headless policies run ACT-R memory-based cognitive algorithms
  - Cognitive parameters come from assets/human_trials_and_cogntive_parameters/CoXAM_cog_param.csv

For each participant/dataset/condition group the script runs a forward episode
and writes per-trial rows. Use --select-best to keep only the lowest-NLL
parameter row per participant.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data_loaders import UnifiedDataLoader
from src.cognitive_models.cr_agent.forward_meta_router import (
    load_forward_strategies,
    run_meta_on_batch,
    COND_DT, COND_LR, COND_DTLR,
)
from src.experiment_planner.cognitive_params import (
    CoXAMCogParams,
    make_memory_factory,
    DEFAULT_TRAINING_COG_PARAMS,
)
from src.experiment_planner.experimental_design import build_design

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = REPO_ROOT / "assets"
DEFAULT_COXAM_PARAMS_PATH = DEFAULT_ASSETS_ROOT / "human_trials_and_cogntive_parameters" / "CoXAM_cog_param.csv"
DEFAULT_META_MODEL_PATH = DEFAULT_ASSETS_ROOT / "weights" / "meta_model.zip"
DEFAULT_DT_MODEL_PATH = DEFAULT_ASSETS_ROOT / "weights" / "dt_policy.zip"
DEFAULT_LR_CALC_MODEL_PATH = DEFAULT_ASSETS_ROOT / "weights" / "lr_calc_policy.zip"
DEFAULT_LR_HEUR_MODEL_PATH = DEFAULT_ASSETS_ROOT / "weights" / "lr_heur_policy.zip"

CSV_COLUMNS = [
    "participant_id",
    "dataset_id",
    "condition",
    "trial_idx",
    "strategy",
    "with_xai",
    "trial_type",
    "prob_correct",
    "pred_time",
    "reward",
    "nll",
    "retrieval_threshold",
    "latency_factor",
    "ddm_a",
    "ddm_s",
    "ddm_Tnd",
    "T_enc",
    "lapse",
    "parameter_row_idx",
]


@dataclass
class CoXAMExperimentConfig:
    """Configuration for generating simulated CoXAM virtual-experiment rows."""

    # PPO model weights
    meta_model_path: str = str(DEFAULT_META_MODEL_PATH)
    dt_model_path: str = str(DEFAULT_DT_MODEL_PATH)
    lr_calc_model_path: str = str(DEFAULT_LR_CALC_MODEL_PATH)
    lr_heur_model_path: str = str(DEFAULT_LR_HEUR_MODEL_PATH)

    # Data and params
    params_path: str = str(DEFAULT_COXAM_PARAMS_PATH)
    assets_root: str = str(DEFAULT_ASSETS_ROOT)

    # Experiment controls
    condition: str = COND_DTLR
    with_xai_ratio: float = 0.5
    chi_value: float = 0.01
    ddm_a_bins: int = 3
    n_trials: int = 36

    # Filtering
    dataset_ids: Optional[List[int]] = None
    max_participants: Optional[int] = None

    # Reproducibility
    seed: int = 7
    select_best: bool = False

    # Optional: pre-built explainers (if None, steps run without XAI context)
    dt_exps: Dict[int, Any] = field(default_factory=dict)
    lr_exps: Dict[int, Any] = field(default_factory=dict)
    training_cog_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TRAINING_COG_PARAMS))


def _load_data(assets_root: str, dataset_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw features and labels for a dataset_id via UnifiedDataLoader."""
    loader = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root=assets_root,
        dataset_id=dataset_id,
    )
    instance_ids = list(range(loader.get_num_instances()))
    features, predictions = loader.get_instances(instance_ids)
    X_raw = np.asarray(features, dtype=float)
    y_raw = np.asarray(predictions, dtype=int)
    return X_raw, y_raw


def _iter_param_rows(
    df_params: pd.DataFrame,
    *,
    dataset_id: int,
    condition: str,
) -> Iterable[Tuple[int, pd.Series]]:
    mask = df_params["dataset_id"].astype(int).eq(dataset_id)
    if "condition" in df_params.columns:
        mask &= df_params["condition"].astype(str).eq(condition)
    return list(df_params[mask].iterrows())


def _iter_participants(
    df_params: pd.DataFrame,
    *,
    dataset_id: int,
    condition: str,
) -> List[str]:
    rows = _iter_param_rows(df_params, dataset_id=dataset_id, condition=condition)
    seen, ordered = set(), []
    for _, row in rows:
        pid = str(row.get("participant_id", row.name))
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def simulate_virtual_experiment(config: CoXAMExperimentConfig) -> pd.DataFrame:
    """Generate simulated CoXAM trial rows as a DataFrame."""
    df_params = pd.read_csv(config.params_path)

    dataset_ids = config.dataset_ids or sorted(df_params["dataset_id"].dropna().astype(int).unique().tolist())
    rng_base = np.random.default_rng(config.seed)

    all_rows: List[Dict[str, Any]] = []

    for dataset_id in dataset_ids:
        try:
            X_raw, y_raw = _load_data(config.assets_root, dataset_id)
        except Exception:
            continue

        N = min(config.n_trials, len(X_raw))
        if N == 0:
            continue

        # Sub-sample X/y to episode length
        idx = rng_base.choice(len(X_raw), size=N, replace=False)
        X_ep = X_raw[idx]
        y_ep = y_raw[idx]

        participant_ids = _iter_participants(df_params, dataset_id=dataset_id, condition=config.condition)
        if config.max_participants is not None:
            participant_ids = participant_ids[: config.max_participants]

        for p_idx, participant_id in enumerate(participant_ids):
            candidate_rows = [
                (row_idx, row)
                for row_idx, row in _iter_param_rows(df_params, dataset_id=dataset_id, condition=config.condition)
                if str(row.get("participant_id", row_idx)) == participant_id
            ]
            if not candidate_rows:
                continue

            cand_results: List[Tuple[int, pd.Series, float, List[Dict[str, Any]]]] = []

            for row_idx, param_row in candidate_rows:
                cog = CoXAMCogParams.from_csv_row(param_row)
                decay = float(param_row.get("decay_param", 0.5)) if not pd.isna(param_row.get("decay_param", 0.5)) else 0.5
                mem_factory = make_memory_factory(decay_param=decay)

                strategies = load_forward_strategies(
                    dt_model_path=config.dt_model_path,
                    lr_calc_model_path=config.lr_calc_model_path,
                    lr_heur_model_path=config.lr_heur_model_path,
                    dt_exps=config.dt_exps,
                    lr_exps=config.lr_exps,
                    memory_factory=mem_factory,
                    training_cog_params=config.training_cog_params,
                    ddm_a_bins=config.ddm_a_bins,
                )

                design = build_design(
                    N=N,
                    condition=config.condition,
                    dataset_id=dataset_id,
                    episode_cogs=cog.to_dict(),
                    with_xai_ratio=config.with_xai_ratio,
                    rng=np.random.default_rng(config.seed + p_idx),
                )

                ep_result = run_meta_on_batch(
                    meta_model=PPO.load(config.meta_model_path),
                    strategies=strategies,
                    X_raw=X_ep,
                    y_raw=y_ep,
                    with_xai_schedule=design.with_xai_schedule,
                    trial_type_schedule=design.trial_type_schedule,
                    condition=config.condition,
                    dataset_id=dataset_id,
                    episode_cogs=cog.to_dict(),
                    training_cog_params=config.training_cog_params,
                    chi_value=config.chi_value,
                    rng=np.random.default_rng(config.seed + p_idx),
                )

                logs = ep_result["logs"]
                probs_correct = logs["prob_correct"]
                losses = [-math.log(max(p, 1e-12)) for p in probs_correct if p > 0]
                nll = float(np.mean(losses)) if losses else float("inf")

                trial_rows = [
                    {
                        "participant_id": participant_id,
                        "dataset_id": dataset_id,
                        "condition": config.condition,
                        "trial_idx": t,
                        "strategy": logs["strategy_name"][t],
                        "with_xai": logs["with_xai_used"][t],
                        "trial_type": logs["trial_type"][t],
                        "prob_correct": logs["prob_correct"][t],
                        "pred_time": logs["pred_time"][t],
                        "reward": logs["reward"][t],
                        "nll": nll,
                        "retrieval_threshold": cog.retrieval_threshold,
                        "latency_factor": cog.latency_factor,
                        "ddm_a": cog.ddm_a,
                        "ddm_s": cog.ddm_s,
                        "ddm_Tnd": cog.ddm_Tnd,
                        "T_enc": cog.T_enc,
                        "lapse": cog.lapse,
                        "parameter_row_idx": row_idx,
                    }
                    for t in range(len(logs["strategy_name"]))
                ]
                cand_results.append((row_idx, param_row, nll, trial_rows))

            if not cand_results:
                continue

            if config.select_best:
                cand_results = [min(cand_results, key=lambda x: x[2])]

            for _, _, _, trial_rows in cand_results:
                all_rows.extend(trial_rows)

    df_out = pd.DataFrame(all_rows)
    for col in CSV_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    return df_out[CSV_COLUMNS] if not df_out.empty else pd.DataFrame(columns=CSV_COLUMNS)


def save_virtual_experiment_results(df: pd.DataFrame, output_path) -> Path:
    """Save simulated CoXAM trial rows to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_COXAM_PARAMS_PATH))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--meta-model", default=str(DEFAULT_META_MODEL_PATH))
    parser.add_argument("--dt-model", default=str(DEFAULT_DT_MODEL_PATH))
    parser.add_argument("--lr-calc-model", default=str(DEFAULT_LR_CALC_MODEL_PATH))
    parser.add_argument("--lr-heur-model", default=str(DEFAULT_LR_HEUR_MODEL_PATH))
    parser.add_argument("--output", default="simulated_results/coxam_simulated_trials.csv")
    parser.add_argument("--condition", default=COND_DTLR, choices=[COND_DT, COND_LR, COND_DTLR])
    parser.add_argument("--with-xai-ratio", type=float, default=0.5)
    parser.add_argument("--chi-value", type=float, default=0.01)
    parser.add_argument("--n-trials", type=int, default=36)
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--select-best", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CoXAMExperimentConfig(
        meta_model_path=args.meta_model,
        dt_model_path=args.dt_model,
        lr_calc_model_path=args.lr_calc_model,
        lr_heur_model_path=args.lr_heur_model,
        params_path=args.params,
        assets_root=args.assets_root,
        condition=args.condition,
        with_xai_ratio=args.with_xai_ratio,
        chi_value=args.chi_value,
        n_trials=args.n_trials,
        max_participants=args.max_participants,
        seed=args.seed,
        select_best=args.select_best,
    )
    df_out = simulate_virtual_experiment(config)
    output_path = save_virtual_experiment_results(df_out, args.output)
    print(f"Wrote {len(df_out)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()

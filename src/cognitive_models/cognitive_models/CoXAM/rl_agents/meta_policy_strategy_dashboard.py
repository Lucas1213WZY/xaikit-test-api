from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import sys

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agents.meta_policy_strategy import (
    CONDITIONS,
    META_STRATEGIES,
    CombinedPolicyConfig,
    CombinedStrategyPolicyEnv,
    load_combined_bundles,
    load_sub_policies,
)


LOW_LEVEL_RUN = ROOT / "outputs" / "unified_strategy_policy" / "unified_strategy_demo"
META_RUN = ROOT / "outputs" / "combined_strategy_meta_policy" / "meta_policy_strategy_masked_explanation_history_demo" / "mixed"
DATASET_ALL = "__all__"
META_POLICY = "meta_policy"


def _default_config() -> CombinedPolicyConfig:
    return CombinedPolicyConfig(
        data_dir=str(ROOT / "datasets"),
        output_root=str(ROOT / "outputs" / "combined_strategy_meta_policy"),
        run_name="combined_strategy_meta_demo",
        condition_name="mixed",
        total_timesteps=5e5,
        n_envs=4,
        instances_per_episode=40,
        max_features=6,
        explanation_shown_ratio=0.5,
        history_window=5,
        simulation_sample_count=16,
        memory_recall_threshold_min=-5.0,
        memory_recall_threshold_max=2.0,
        unavailable_strategy_penalty=-1.0,
        apps=None,
        lr_calculation_model_path=str(LOW_LEVEL_RUN / "lr_calculation" / "models" / "final_model.zip"),
        lr_heuristic_model_path=str(LOW_LEVEL_RUN / "lr_heuristic" / "models" / "final_model.zip"),
        dt_traversal_model_path=str(LOW_LEVEL_RUN / "dt_traversal" / "models" / "final_model.zip"),
    )


def _strategy_available(condition_name: str, strategy_name: str) -> bool:
    if condition_name == "linear_regression":
        return strategy_name in {"lr_calculation", "lr_heuristic"}
    if condition_name == "decision_tree":
        return strategy_name == "dt_traversal"
    return True


def _available_policies(condition_name: str) -> list[str]:
    policies = [META_POLICY]
    if condition_name != "mixed":
        policies.extend(
            strategy_name
            for strategy_name in META_STRATEGIES
            if _strategy_available(condition_name, strategy_name)
        )
    return policies


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


class ExperimentService:
    def __init__(self) -> None:
        self.base_config = _default_config()
        self.meta_model_path = META_RUN / "models" / "best_model.zip"
        if not self.meta_model_path.exists():
            raise FileNotFoundError(f"Missing meta-policy model: {self.meta_model_path}")
        try:
            self.model = MaskablePPO.load(self.meta_model_path, device="cpu")
            self.uses_action_masks = True
        except ValueError:
            self.model = PPO.load(self.meta_model_path, device="cpu")
            self.uses_action_masks = False
            self.base_config = CombinedPolicyConfig(
                **{
                    **asdict(self.base_config),
                    "randomize_strategy_order_per_episode": False,
                    "meta_history_by_explanation": False,
                }
            )
        self.bundles = load_combined_bundles(ROOT / "datasets", self.base_config)
        self.sub_policies = load_sub_policies(self.base_config)

    def options(self) -> dict[str, Any]:
        datasets = [
            {
                "id": f"{bundle.app_id}||{bundle.model_name}",
                "label": f"{bundle.app_id} / {bundle.model_name}",
                "app_id": bundle.app_id,
                "model_name": bundle.model_name,
                "instances": len(bundle.instance_ids),
            }
            for bundle in self.bundles
        ]
        return {
            "datasets": datasets,
            "conditions": ["mixed", *CONDITIONS],
            "strategies": list(META_STRATEGIES),
            "policies_by_condition": {
                condition: _available_policies(condition)
                for condition in ["mixed", *CONDITIONS]
            },
            "defaults": {
                "dataset": DATASET_ALL,
                "condition": "mixed",
                "policy": META_POLICY,
                "repeats": 5,
                "episodes_per_repeat": 1,
                "instances_per_episode": self.base_config.instances_per_episode,
                "decision_noise": 0.4,
                "memory_recall_threshold": 0.5,
                "opportunity_cost": 0.01,
                "explanation_shown_ratio": self.base_config.explanation_shown_ratio,
                "target_xai_fidelity": 0.9,
                "simulation_sample_count": self.base_config.simulation_sample_count,
                "retrieval_candidate_count": self.base_config.retrieval_candidate_count,
                "seed": self.base_config.seed,
            },
        }

    def _selected_bundles(self, dataset_id: str):
        if dataset_id == DATASET_ALL:
            return self.bundles
        app_id, model_name = dataset_id.split("||", 1)
        return [b for b in self.bundles if b.app_id == app_id and b.model_name == model_name]

    def _feature_labels(self, dataset_id: str, bundles) -> dict[int, str]:
        labels = {i: f"a{i}" for i in range(self.base_config.max_features)}
        if dataset_id == DATASET_ALL or not bundles:
            return labels
        names = bundles[0].loader.get_feature_names(bundles[0].app_id)
        for i in range(self.base_config.max_features):
            labels[i] = names.get(f"a{i}", f"a{i}")
        return labels

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload.get("dataset", DATASET_ALL))
        condition_name = str(payload.get("condition", "mixed"))
        policy_name = str(payload.get("policy", META_POLICY))
        available_policies = _available_policies(condition_name)
        if policy_name not in available_policies:
            raise ValueError(
                f"Policy {policy_name!r} is not available for condition {condition_name!r}; "
                f"expected one of {available_policies}."
            )
        repeats = max(1, min(20, int(payload.get("repeats", 5))))
        episodes_per_repeat = max(1, min(20, int(payload.get("episodes_per_repeat", 1))))
        instances_per_episode = max(4, min(120, int(payload.get("instances_per_episode", 40))))
        seed = int(payload.get("seed", self.base_config.seed))

        bundles = self._selected_bundles(dataset_id)
        if not bundles:
            raise ValueError(f"No dataset bundle found for {dataset_id!r}")

        config = CombinedPolicyConfig(
            **{
                **asdict(self.base_config),
                "condition_name": condition_name,
                "instances_per_episode": instances_per_episode,
                "explanation_shown_ratio": float(payload.get("explanation_shown_ratio", 0.5)),
                "target_xai_fidelity": float(np.clip(float(payload.get("target_xai_fidelity", 0.9)), 0.0, 1.0)),
                "simulation_sample_count": int(payload.get("simulation_sample_count", 16)),
                "retrieval_candidate_count": int(payload.get("retrieval_candidate_count", 3)),
            }
        )
        fixed_eval_params = {
            "decision_noise": float(payload.get("decision_noise", 0.4)),
            "memory_recall_threshold": float(payload.get("memory_recall_threshold", 0.5)),
            "opportunity_cost": float(payload.get("opportunity_cost", 0.01)),
        }

        started = time.time()
        env = CombinedStrategyPolicyEnv(
            bundles,
            config,
            self.sub_policies,
            training=False,
            fixed_eval_params=fixed_eval_params,
        )
        rows: list[dict[str, Any]] = []
        seed_rng = np.random.default_rng(seed)
        for repeat in range(repeats):
            for episode in range(episodes_per_repeat):
                episode_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
                obs, _ = env.reset(seed=episode_seed)
                done = False
                while not done:
                    if policy_name != META_POLICY:
                        action = env.strategy_slots.index(policy_name)
                    elif self.uses_action_masks:
                        action, _ = self.model.predict(obs, deterministic=True, action_masks=env.action_masks())
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rows.append(
                        {
                            "repeat": repeat,
                            "episode": episode,
                            "episode_seed": episode_seed,
                            "step": env.step_idx - 1,
                            "reward": float(reward),
                            **info,
                        }
                    )
                    done = bool(terminated or truncated)
        env.close()

        return {
            "settings": {
                "dataset": dataset_id,
                "condition": condition_name,
                "policy": policy_name,
                "repeats": repeats,
                "episodes_per_repeat": episodes_per_repeat,
                "instances_per_episode": instances_per_episode,
                "target_xai_fidelity": config.target_xai_fidelity,
                **fixed_eval_params,
            },
            "elapsed_seconds": time.time() - started,
            "row_count": len(rows),
            "summaries": self._summarize(rows, instances_per_episode, self._feature_labels(dataset_id, bundles)),
        }

    def _summarize(self, rows: list[dict[str, Any]], instances_per_episode: int, feature_labels: dict[int, str]) -> dict[str, Any]:
        if not rows:
            return {}
        df = pd.DataFrame(rows)
        df["trial_quarter"] = np.minimum((df["step"].astype(int) * 4) // max(1, instances_per_episode), 3).astype(int)
        quarter_labels = {0: "0-25%", 1: "25-50%", 2: "50-75%", 3: "75-100%"}
        df["trial_window"] = df["trial_quarter"].map(quarter_labels)

        selection_parts = []
        for repeat, repeat_df in df.groupby("repeat"):
            totals = (
                repeat_df.groupby(["condition_name", "explanation_type", "trial_quarter"], dropna=False)
                .size()
                .reset_index(name="total")
            )
            counts = (
                repeat_df.groupby(["condition_name", "explanation_type", "trial_quarter", "selected_strategy"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            base_rows = []
            for total_row in totals.itertuples(index=False):
                for strategy_name in META_STRATEGIES:
                    base_rows.append(
                        {
                            "repeat": repeat,
                            "condition_name": total_row.condition_name,
                            "explanation_type": total_row.explanation_type,
                            "trial_quarter": total_row.trial_quarter,
                            "selected_strategy": strategy_name,
                        }
                    )
            base = pd.DataFrame(base_rows)
            merged = base.merge(
                counts,
                on=["condition_name", "explanation_type", "trial_quarter", "selected_strategy"],
                how="left",
            )
            merged = merged.merge(totals, on=["condition_name", "explanation_type", "trial_quarter"], how="left")
            merged["selection_rate"] = merged["count"].fillna(0) / merged["total"].clip(lower=1)
            selection_parts.append(merged)
        selection_df = pd.concat(selection_parts, ignore_index=True)
        strategy_by_window = (
            selection_df.groupby(
                ["condition_name", "explanation_type", "trial_quarter", "selected_strategy"],
                dropna=False,
            )
            .agg(
                mean_selection_rate=("selection_rate", "mean"),
                min_selection_rate=("selection_rate", "min"),
                max_selection_rate=("selection_rate", "max"),
            )
            .reset_index()
        )
        strategy_by_window["trial_window"] = strategy_by_window["trial_quarter"].map(quarter_labels)

        for idx in range(len(feature_labels)):
            df[f"feature_{idx}_selected"] = df["selected_feature_mask"].apply(
                lambda mask, i=idx: int(i < len(mask) and int(mask[i]) == 1)
            )
        linear_df = df[df["selected_strategy"].isin(["lr_calculation", "lr_heuristic"])].copy()
        feature_rows = []
        if not linear_df.empty:
            for (condition_name, quarter), group in linear_df.groupby(
                ["condition_name", "trial_quarter"], dropna=False
            ):
                for idx in range(len(feature_labels)):
                    feature_rows.append(
                        {
                            "condition_name": condition_name,
                            "trial_quarter": int(quarter),
                            "trial_window": quarter_labels[int(quarter)],
                            "feature": f"a{idx}",
                            "feature_label": feature_labels.get(idx, f"a{idx}"),
                            "selection_rate": float(group[f"feature_{idx}_selected"].mean()),
                            "eligible_trials": int(len(group)),
                        }
                    )

        strategy_metrics = []
        for strategy_name in META_STRATEGIES:
            available = df["condition_name"].apply(lambda condition: _strategy_available(str(condition), strategy_name))
            available_df = df[available]
            if available_df.empty:
                continue
            selected_df = available_df[available_df["selected_strategy"] == strategy_name]
            selected_accuracy = (
                float(selected_df[f"{strategy_name}_prob_correct"].mean())
                if not selected_df.empty
                else 0.0
            )
            strategy_metrics.append(
                {
                    "strategy": strategy_name,
                    "available_trials": int(len(available_df)),
                    "selection_rate": float((available_df["selected_strategy"] == strategy_name).mean()),
                    "accuracy": selected_accuracy,
                    "mean_prob_correct": float(available_df[f"{strategy_name}_prob_correct"].mean()),
                    "mean_time": float(available_df[f"{strategy_name}_pred_time"].mean()),
                }
            )

        repeat_summary = (
            df.groupby("repeat", dropna=False)
            .agg(
                trials=("reward", "size"),
                seed=("episode_seed", "first"),
                accuracy=("selected_prob_correct", "mean"),
                mean_prob_correct=("selected_prob_correct", "mean"),
                mean_time=("selected_pred_time", "mean"),
                mean_reward=("reward", "mean"),
                lr_xai_fidelity=("lr_explanation_matches_ai", "mean"),
                dt_xai_fidelity=("dt_explanation_matches_ai", "mean"),
            )
            .reset_index()
        )
        overall = {
            "trials": int(df.shape[0]),
            "accuracy": float(repeat_summary["accuracy"].mean()),
            "accuracy_std": float(repeat_summary["accuracy"].std(ddof=0)),
            "mean_prob_correct": float(repeat_summary["mean_prob_correct"].mean()),
            "mean_time": float(repeat_summary["mean_time"].mean()),
            "mean_time_std": float(repeat_summary["mean_time"].std(ddof=0)),
            "mean_reward": float(repeat_summary["mean_reward"].mean()),
            "lr_xai_fidelity": float(repeat_summary["lr_xai_fidelity"].mean()),
            "dt_xai_fidelity": float(repeat_summary["dt_xai_fidelity"].mean()),
        }

        return {
            "strategy_by_window": strategy_by_window.to_dict(orient="records"),
            "linear_feature_by_window": feature_rows,
            "strategy_metrics": strategy_metrics,
            "overall": overall,
            "repeat_summary": repeat_summary.to_dict(orient="records"),
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meta Policy Strategy Dashboard</title>
  <style>
    :root { color-scheme: light; --blue:#4c78a8; --orange:#f58518; --green:#54a24b; --ink:#17202a; --muted:#667085; --line:#d0d5dd; --bg:#f6f7f9; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); background:var(--bg); }
    .app { display:grid; grid-template-columns:320px 1fr; min-height:100vh; }
    aside { background:#fff; border-right:1px solid var(--line); padding:18px; position:sticky; top:0; height:100vh; overflow:auto; }
    main { padding:20px 24px 36px; }
    h1 { font-size:20px; margin:0 0 4px; }
    h2 { font-size:16px; margin:24px 0 10px; }
    label { display:block; font-weight:650; margin-top:14px; }
    select, input { width:100%; padding:8px 9px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); }
    input[type=range] { padding:0; }
    .value { color:var(--muted); float:right; font-weight:500; }
    button { width:100%; margin-top:18px; padding:10px 12px; border:0; border-radius:6px; background:#1f2937; color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .subtle { color:var(--muted); font-size:12px; }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)); gap:12px; }
    .card, .panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; }
    .metric { font-size:24px; font-weight:750; margin-top:4px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }
    table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    th, td { text-align:left; border-bottom:1px solid #eaecf0; padding:8px 9px; }
    th { font-size:12px; color:var(--muted); background:#f9fafb; }
    tr:last-child td { border-bottom:0; }
    .chart-title { font-weight:750; margin:2px 0 8px; }
    svg { width:100%; height:auto; display:block; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; margin:8px 0; }
    .swatch { width:10px; height:10px; display:inline-block; border-radius:2px; margin-right:5px; }
    .empty { padding:24px; color:var(--muted); text-align:center; border:1px dashed var(--line); border-radius:8px; background:#fff; }
    @media (max-width: 980px) { .app { grid-template-columns:1fr; } aside { position:relative; height:auto; } .grid, .cards { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Meta Policy Strategy</h1>
    <div class="subtle">Runs the meta policy or a compatible individual strategy policy.</div>
    <label>Dataset<select id="dataset"></select></label>
    <label>Condition<select id="condition"></select></label>
    <label>Policy to run<select id="policy"></select></label>
    <label>Repeats <span class="value" id="repeatsValue"></span><input id="repeats" type="range" min="1" max="10" step="1"></label>
    <label>Episodes per repeat <span class="value" id="episodesValue"></span><input id="episodes_per_repeat" type="range" min="1" max="8" step="1"></label>
    <label>Trials per episode <span class="value" id="instancesValue"></span><input id="instances_per_episode" type="range" min="12" max="80" step="4"></label>
    <label>Decision noise <span class="value" id="noiseValue"></span><input id="decision_noise" type="range" min="0.3" max="0.5" step="0.01"></label>
    <label>Memory recall threshold <span class="value" id="memoryValue"></span><input id="memory_recall_threshold" type="range" min="-5" max="2" step="0.05"></label>
    <label>Opportunity cost <span class="value" id="costValue"></span><input id="opportunity_cost" type="range" min="0" max="0.02" step="0.001"></label>
    <label>Explanation shown ratio <span class="value" id="xaiValue"></span><input id="explanation_shown_ratio" type="range" min="0" max="1" step="0.05"></label>
    <label>XAI fidelity target <span class="value" id="fidelityValue"></span><input id="target_xai_fidelity" type="range" min="0" max="1" step="0.025"></label>
    <label>Simulation samples <span class="value" id="samplesValue"></span><input id="simulation_sample_count" type="range" min="4" max="64" step="4"></label>
    <label>Retrieval candidates <span class="value" id="candidatesValue"></span><input id="retrieval_candidate_count" type="range" min="1" max="8" step="1"></label>
    <label>Seed<input id="seed" type="number"></label>
    <button id="run">Run Experiments</button>
    <p class="subtle" id="status">Ready.</p>
  </aside>
  <main>
    <div class="cards" id="cards"></div>
    <h2>Strategy Selection Over Trials</h2>
    <div class="legend" id="legend"></div>
    <div id="selectionCharts" class="grid"></div>
    <h2>Linear Feature Selection Over Time</h2>
    <div id="featureCharts" class="grid"></div>
    <h2>Strategy Accuracy And Time</h2>
    <div id="strategyTable"></div>
    <h2>Repeat Averages</h2>
    <div id="repeatTable"></div>
  </main>
</div>
<script>
const STRATEGIES = ["lr_calculation", "lr_heuristic", "dt_traversal"];
const COLORS = {lr_calculation:"#4c78a8", lr_heuristic:"#f58518", dt_traversal:"#54a24b"};
const LABELS = {lr_calculation:"LR calculation", lr_heuristic:"LR heuristic", dt_traversal:"DT traversal"};
const POLICY_LABELS = {meta_policy:"Meta policy", ...LABELS};
const EXPLANATION_LABELS = {none:"No explanation", lr:"Linear regression explanation", dt:"Decision tree explanation"};
const EXPLANATION_ORDER = {none:0, dt:1, lr:2};
const $ = id => document.getElementById(id);
const fmt = (v, d=3) => Number.isFinite(+v) ? (+v).toFixed(d) : "0";
let policiesByCondition = {};

function bindValue(inputId, labelId, digits=3) {
  const input = $(inputId), label = $(labelId);
  const update = () => label.textContent = input.type === "range" ? fmt(input.value, digits) : input.value;
  input.addEventListener("input", update); update();
}

async function loadOptions() {
  const res = await fetch("/api/options");
  const options = await res.json();
  $("dataset").innerHTML = `<option value="__all__">All datasets</option>` + options.datasets.map(d => `<option value="${d.id}">${d.label}</option>`).join("");
  $("condition").innerHTML = options.conditions.map(c => `<option value="${c}">${c}</option>`).join("");
  policiesByCondition = options.policies_by_condition || {};
  for (const [key, value] of Object.entries(options.defaults)) if ($(key)) $(key).value = value;
  updatePolicyOptions(options.defaults.policy);
  $("condition").addEventListener("change", () => updatePolicyOptions());
  bindValue("repeats", "repeatsValue", 0);
  bindValue("episodes_per_repeat", "episodesValue", 0);
  bindValue("instances_per_episode", "instancesValue", 0);
  bindValue("decision_noise", "noiseValue", 2);
  bindValue("memory_recall_threshold", "memoryValue", 2);
  bindValue("opportunity_cost", "costValue", 3);
  bindValue("explanation_shown_ratio", "xaiValue", 2);
  bindValue("target_xai_fidelity", "fidelityValue", 3);
  bindValue("simulation_sample_count", "samplesValue", 0);
  bindValue("retrieval_candidate_count", "candidatesValue", 0);
  renderLegend();
}

function updatePolicyOptions(preferredPolicy) {
  const policies = policiesByCondition[$("condition").value] || ["meta_policy"];
  const current = preferredPolicy || $("policy").value;
  $("policy").innerHTML = policies.map(policy =>
    `<option value="${policy}">${POLICY_LABELS[policy] || policy}</option>`
  ).join("");
  $("policy").value = policies.includes(current) ? current : "meta_policy";
}

function payload() {
  return {
    dataset: $("dataset").value,
    condition: $("condition").value,
    policy: $("policy").value,
    repeats: +$("repeats").value,
    episodes_per_repeat: +$("episodes_per_repeat").value,
    instances_per_episode: +$("instances_per_episode").value,
    decision_noise: +$("decision_noise").value,
    memory_recall_threshold: +$("memory_recall_threshold").value,
    opportunity_cost: +$("opportunity_cost").value,
    explanation_shown_ratio: +$("explanation_shown_ratio").value,
    target_xai_fidelity: +$("target_xai_fidelity").value,
    simulation_sample_count: +$("simulation_sample_count").value,
    retrieval_candidate_count: +$("retrieval_candidate_count").value,
    seed: +$("seed").value
  };
}

async function runExperiment() {
  $("run").disabled = true;
  $("status").textContent = "Running experiments...";
  try {
    const res = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload())});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Experiment failed");
    render(data);
    $("status").textContent = `Done: ${data.row_count} trials in ${fmt(data.elapsed_seconds, 1)}s.`;
  } catch (err) {
    $("status").textContent = err.message;
  } finally {
    $("run").disabled = false;
  }
}

function renderLegend() {
  $("legend").innerHTML = STRATEGIES.map(s => `<span><i class="swatch" style="background:${COLORS[s]}"></i>${LABELS[s]}</span>`).join("");
}

function renderCards(overall={}) {
  const cards = [
    ["Trials", overall.trials || 0, 0],
    ["Overall accuracy", overall.accuracy || 0, 3],
    ["Mean time", overall.mean_time || 0, 2],
    ["Mean reward", overall.mean_reward || 0, 3],
    ["LR XAI fidelity", overall.lr_xai_fidelity || 0, 3],
    ["DT XAI fidelity", overall.dt_xai_fidelity || 0, 3],
  ];
  $("cards").innerHTML = cards.map(([name, val, digits]) => `<div class="card"><div class="subtle">${name}</div><div class="metric">${fmt(val, digits)}</div></div>`).join("");
}

function conditionsFrom(rows) {
  return [...new Set(rows.map(r => r.condition_name))].sort();
}

function renderSelectionCharts(rows) {
  const root = $("selectionCharts");
  if (!rows.length) { root.innerHTML = `<div class="empty">No strategy selection rows.</div>`; return; }
  const groups = [...new Map(rows.map(r => [`${r.condition_name}||${r.explanation_type}`, r])).values()]
    .sort((a, b) => a.condition_name.localeCompare(b.condition_name) ||
      (EXPLANATION_ORDER[a.explanation_type] ?? 99) - (EXPLANATION_ORDER[b.explanation_type] ?? 99));
  root.innerHTML = groups.map(group => {
    const subset = rows.filter(r => r.condition_name === group.condition_name && r.explanation_type === group.explanation_type);
    const explanation = EXPLANATION_LABELS[group.explanation_type] || group.explanation_type;
    return `<div class="panel"><div class="chart-title">${group.condition_name} / ${explanation}</div>${stackedBars(subset)}</div>`;
  }).join("");
}

function stackedBars(rows) {
  const W=520, H=260, left=42, top=14, bottom=34, barW=70, gap=34;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  svg += `<line x1="${left}" y1="${H-bottom}" x2="${W-12}" y2="${H-bottom}" stroke="#d0d5dd"/>`;
  for (let q=0; q<4; q++) {
    let y = H-bottom, x = left + q*(barW+gap) + 24;
    for (const s of STRATEGIES) {
      const r = rows.find(v => +v.trial_quarter === q && v.selected_strategy === s);
      const h = ((r ? +r.mean_selection_rate : 0) * (H-top-bottom));
      y -= h;
      svg += `<rect x="${x}" y="${y}" width="${barW}" height="${Math.max(0,h)}" fill="${COLORS[s]}"><title>${LABELS[s]} ${fmt(r?.mean_selection_rate || 0, 2)}</title></rect>`;
    }
    svg += `<text x="${x+barW/2}" y="${H-10}" text-anchor="middle" font-size="12">${["0-25%","25-50%","50-75%","75-100%"][q]}</text>`;
  }
  for (let t=0; t<=1.001; t+=0.25) {
    const y = H-bottom - t*(H-top-bottom);
    svg += `<line x1="${left-4}" y1="${y}" x2="${W-12}" y2="${y}" stroke="#eaecf0"/><text x="8" y="${y+4}" font-size="11" fill="#667085">${fmt(t,2)}</text>`;
  }
  return svg + `</svg>`;
}

function renderFeatureCharts(rows) {
  const root = $("featureCharts");
  if (!rows.length) { root.innerHTML = `<div class="empty">No linear strategy selections in this run.</div>`; return; }
  root.innerHTML = conditionsFrom(rows).map(condition => {
    const subset = rows.filter(r => r.condition_name === condition);
    return `<div class="panel"><div class="chart-title">${condition}</div>${heatmap(subset)}</div>`;
  }).join("");
}

function heatColor(v) {
  const x = Math.max(0, Math.min(1, +v || 0));
  const a = Math.round(245 - 120*x), b = Math.round(247 - 110*x), c = Math.round(249 - 60*x);
  return `rgb(${a},${b},${c})`;
}

function heatmap(rows) {
  const features = [...new Map(rows.map(r => [r.feature, r.feature_label])).entries()];
  const cellW=82, cellH=24, left=150, top=18, W=left + 4*cellW + 20, H=top + features.length*cellH + 34;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let q=0; q<4; q++) svg += `<text x="${left+q*cellW+cellW/2}" y="12" text-anchor="middle" font-size="11" fill="#667085">${["0-25%","25-50%","50-75%","75-100%"][q]}</text>`;
  features.forEach(([feature, label], i) => {
    const y = top + i*cellH;
    svg += `<text x="${left-8}" y="${y+16}" text-anchor="end" font-size="11">${label}</text>`;
    for (let q=0; q<4; q++) {
      const r = rows.find(v => v.feature === feature && +v.trial_quarter === q);
      const val = r ? +r.selection_rate : 0;
      svg += `<rect x="${left+q*cellW}" y="${y}" width="${cellW-2}" height="${cellH-2}" fill="${heatColor(val)}" stroke="#fff"><title>${label}: ${fmt(val,2)}</title></rect>`;
      svg += `<text x="${left+q*cellW+cellW/2}" y="${y+15}" text-anchor="middle" font-size="10">${fmt(val,2)}</text>`;
    }
  });
  return svg + `</svg>`;
}

function table(rows, columns) {
  if (!rows.length) return `<div class="empty">No rows.</div>`;
  return `<table><thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(r => `<tr>${columns.map(c => `<td>${c.format ? c.format(r[c.key], r) : r[c.key]}</td>`).join("")}</tr>`).join("") +
    `</tbody></table>`;
}

function renderTables(summary) {
  $("strategyTable").innerHTML = table(summary.strategy_metrics || [], [
    {key:"strategy", label:"Strategy", format:v => LABELS[v] || v},
    {key:"selection_rate", label:"Selected", format:v => fmt(v,3)},
    {key:"accuracy", label:"Selected accuracy", format:v => fmt(v,3)},
    {key:"mean_prob_correct", label:"Available P(correct)", format:v => fmt(v,3)},
    {key:"mean_time", label:"Mean time", format:v => fmt(v,2)},
    {key:"available_trials", label:"Available trials"}
  ]);
  $("repeatTable").innerHTML = table(summary.repeat_summary || [], [
    {key:"repeat", label:"Repeat"},
    {key:"seed", label:"Seed"},
    {key:"trials", label:"Trials"},
    {key:"lr_xai_fidelity", label:"LR XAI fidelity", format:v => fmt(v,3)},
    {key:"dt_xai_fidelity", label:"DT XAI fidelity", format:v => fmt(v,3)},
    {key:"accuracy", label:"Accuracy", format:v => fmt(v,3)},
    {key:"mean_prob_correct", label:"P(correct)", format:v => fmt(v,3)},
    {key:"mean_time", label:"Mean time", format:v => fmt(v,2)},
    {key:"mean_reward", label:"Reward", format:v => fmt(v,3)}
  ]);
}

function render(data) {
  const s = data.summaries || {};
  renderCards(s.overall || {});
  renderSelectionCharts(s.strategy_by_window || []);
  renderFeatureCharts(s.linear_feature_by_window || []);
  renderTables(s);
}

$("run").addEventListener("click", runExperiment);
loadOptions().then(() => renderCards({}));
</script>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    service: ExperimentService

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/options":
            self._send_json(self.service.options())
            return
        if path in {"/", "/index.html"}:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._send_json(self.service.run(payload))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the meta-policy strategy dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    RequestHandler.service = ExperimentService()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Meta policy strategy dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

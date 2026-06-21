"""
Example: Generate a counterbalanced experiment trial sequence
for an XAI user study using wine quality explanation data.

Output cols per trial:
    participantId | trialId | block | trialWithinBlock | withinCondition
    | display_type (between IV) | dataId | instanceId | modelName
    | expMethod | pred | CV1_model | CV2_dataset

id_map modes demonstrated:
    Mode 1 — dict map: pull dataId/instanceId from named CSV columns
    Mode 2 — auto fallback: column not found → sequential integers
    Mode 3 — None (default): use "dataId"/"instanceId" keys if present, else auto
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiment_design.counterbalance import run_experiment_design

# ── Load instance pool from explanation CSV ────────────────────────────────────
def load_instances(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

instance_pool = load_instances(
    "tutorials/generated_explanation/de_mlp_wine_quality.csv"
)

# ── Define Independent Variables ───────────────────────────────────────────────
iv_config = {
    # Within-subjects: every participant sees all 3 XAI methods
    # → counterbalancing applied (complete: 3! = 6 orders)
    "xai_method": {
        "type": "within",
        "levels": ["shap", "lime", "decision_explanation"],
    },
    # Between-subjects: each participant sees only ONE display type
    # → no counterbalancing needed, round-robin group assignment
    "display_type": {
        "type": "between",
        "levels": ["bar_chart", "heatmap"],
    },
}

# ── Controlled Variables (fixed for all trials, appended as CV cols) ───────────
controlled_vars = {
    "CV1_model":   "mlp",
    "CV2_dataset": "wine_quality",
}

# ── id_map: tells the pipeline which instance columns map to dataId/instanceId ─
#
#   Option A — map to real CSV columns (recommended when your pool has them):
id_map = {"dataId": "dataId", "instanceId": "instanceId"}
#
#   Option B — map to non-existent columns → auto-generates sequential integers:
# id_map = {"dataId": "no_such_col", "instanceId": "no_such_col"}
#
#   Option C — None → tries default "dataId"/"instanceId" keys, auto if absent:
# id_map = None

# ── Run full pipeline ──────────────────────────────────────────────────────────
result = run_experiment_design(
    iv_config=iv_config,
    instance_pool=instance_pool,
    n_participants=12,          # multiple of n_orders (6) keeps assignment balanced
    trials_per_condition=3,     # data instances shown per XAI method block
    controlled_vars=controlled_vars,
    id_map=id_map,
    output_dir="tutorials/experiment_output",
    shuffle_instances=True,
    seed=42,
)

# ── Design summary ─────────────────────────────────────────────────────────────
print(f"\nStrategy:             {result['strategy']}")
print(f"Counterbalance orders: {len(result['orders'])}")
print(f"Total trial rows:      {len(result['trials'])}")

print(f"\nAll factorial conditions ({len(result['all_conditions'])} total):")
for c in result["all_conditions"]:
    print(" ", c)

print("\nParticipant → condition order:")
for a in result["assignments"]:
    print(f"  P{a['participantId']:02d} [{a.get('display_type','')}]  {a['within_order']}")

# ── Sample trial rows ──────────────────────────────────────────────────────────
print("\nSample trial rows (first 3):")
for trial in result["trials"][:3]:
    print({k: trial[k] for k in
           ["participantId", "trialId", "block", "trialWithinBlock",
            "withinCondition", "display_type", "dataId", "instanceId",
            "CV1_model", "CV2_dataset"]})

print(f"\nExported files:")
print(f"  CSV:     {result['csv_path']}")
print(f"  JSON:    {result['json_path']}")
print(f"  Summary: {result['summary_path']}")

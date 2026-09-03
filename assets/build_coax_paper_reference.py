"""Transcribe the published CoAX accuracy tables into one JSON reference.

The CoAX paper's numbers live in three spreadsheets under
``assets/human_data/CoAX/paper/``. This builder folds them into a single JSON
the server can serve without depending on ``openpyxl`` at request time, matching
how every other comparison asset in this directory is produced.

Two tables are read:

* ``All Datasets Accuracy Summary.xlsx`` -- one row per
  ``appId x XAIType x Tested w/ XAI x Agent``, the per-dataset breakdown.
* ``Human CoAX Aggregate Across All Datasets.xlsx`` -- the same cells pooled
  over datasets, which is the paper's headline table.

Only the ``Human`` and ``CoAX`` agents are kept. The published ``DT``/``KNN``/
``MLP`` rows are a different comparison (how well an ML proxy predicts the AI,
not how a person does) and carrying them would treble every panel.

What the numbers mean, and why they are worth comparing a simulation against:
``Mean(Accuracy 2)`` is forward-simulation accuracy -- agreement with the AI's
own prediction -- for the tested block. The ``Human`` rows are the study
participants; the ``CoAX`` rows are 150 simulated agents per cell. Averaging the
fitted table's own ``PAI`` and ``MAI`` columns
(``assets/human_data/CoAX/coax_fitted_strategies.csv``) reproduces these two
series to within about .03 and .02 respectively, which is the evidence that the
shipped fits and these published tables describe the same experiment.

Run:  python assets/build_coax_paper_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.virtual_experiment_executor.participant_pools import (  # noqa: E402
    canon_coax_xai_type,
    canon_dataset,
    canon_tested,
)

PAPER_DIR = REPO_ROOT / "assets" / "human_data" / "CoAX" / "paper"
PER_DATASET_FILE = PAPER_DIR / "All Datasets Accuracy Summary.xlsx"
POOLED_FILE = PAPER_DIR / "Human CoAX Aggregate Across All Datasets.xlsx"
OUTPUT = REPO_ROOT / "assets" / "coax_paper_reference.json"

#: The paper writes dataset names long ("Adult Income"); every table in this
#: repository writes them short. Kept here rather than in ``canon_dataset``
#: because it is a quirk of these three files, and widening the shared
#: canonicalizer would change what every existing pool draw matches.
APP_ID_TO_DATA_ID = {
    "adult_income": "adult",
    "forest_cover_type": "forest_cover",
    "mushrooms": "mushrooms",
    "wine_quality": "wine_quality",
}

#: Display names, so the figure does not have to un-slug them again.
DATA_ID_LABELS = {
    "adult": "Adult income",
    "forest_cover": "Forest cover",
    "mushrooms": "Mushrooms",
    "wine_quality": "Wine quality",
}

AGENTS = {"Human": "human", "CoAX": "coax"}

MEAN = "Mean(Accuracy 2)"
SEM = "Std Err(Accuracy 2)"
LOWER = "Lower 95% CI"
UPPER = "Upper 95% CI"


def _data_id(app_id: Any) -> str:
    text = canon_dataset(app_id)
    return APP_ID_TO_DATA_ID.get(text, text)


def _cells(frame: pd.DataFrame, *, per_dataset: bool) -> list[dict[str, Any]]:
    """One record per published cell, canonicalized onto this repo's spellings."""
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        agent = AGENTS.get(str(row["Agent"]).strip())
        if agent is None:
            continue
        # The `none` arm shows no explanation on any trial, so its "tested with
        # XAI" half does not exist -- the published rows for it are all-NaN
        # placeholders. Mirrors COAX_IMPOSSIBLE_CELLS in coax_trial_executor.
        if pd.isna(row[MEAN]):
            continue
        record: dict[str, Any] = {
            "xai_type": canon_coax_xai_type(row["XAIType"]),
            "tested_w_xai": canon_tested(row["Tested w/ XAI"]) == "true",
            "agent": agent,
            "n": int(row["N Rows"]),
            "mean": float(row[MEAN]),
            "sem": None if pd.isna(row[SEM]) else float(row[SEM]),
            "ci_lower": None if pd.isna(row[LOWER]) else float(row[LOWER]),
            "ci_upper": None if pd.isna(row[UPPER]) else float(row[UPPER]),
        }
        if per_dataset:
            data_id = _data_id(row["appId"])
            record["dataId"] = data_id
            record["dataset_label"] = DATA_ID_LABELS.get(data_id, data_id)
        rows.append(record)
    return rows


def build() -> dict[str, Any]:
    per_dataset = _cells(pd.read_excel(PER_DATASET_FILE), per_dataset=True)
    pooled = _cells(pd.read_excel(POOLED_FILE), per_dataset=False)

    datasets = sorted({row["dataId"] for row in per_dataset})
    return {
        "generated_from": "assets/build_coax_paper_reference.py",
        "sources": [
            str(PER_DATASET_FILE.relative_to(REPO_ROOT)),
            str(POOLED_FILE.relative_to(REPO_ROOT)),
        ],
        "dv": "Forward simulation accuracy",
        "dv_note": (
            "Agreement with the AI's own prediction on the tested block. "
            "'human' is the study participants; 'coax' is the published "
            "simulation, 150 agents per cell."
        ),
        "datasets": datasets,
        "dataset_labels": {d: DATA_ID_LABELS.get(d, d) for d in datasets},
        "per_dataset": per_dataset,
        "pooled": pooled,
    }


def main() -> None:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  per-dataset cells: {len(payload['per_dataset'])}")
    print(f"  pooled cells:      {len(payload['pooled'])}")
    print(f"  datasets:          {', '.join(payload['datasets'])}")


if __name__ == "__main__":
    main()

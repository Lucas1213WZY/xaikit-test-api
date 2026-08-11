"""Build the NLL/BIC model-fit comparison tables for the UI to render.

Emits one JSON (``assets/human_vs_model_fit_stats.json``) with, per study, one
row per model (the study's own cognitive model plus any machine-proxy
baselines) and one NLL/BIC cell per facet (``exp_property`` for sim2real).
Mean +/- SD is across participants, matching the three LaTeX tables the UI
needs to reproduce (CoXAM forward, CoXAM counterfactual, CoAX forward).

Only sim2real has a baseline NLL/BIC comparison today --
``fit_sim2real_baselines_to_participants.py`` fits DT/logistic-regression/MLP
proxies and scores them against real participant answers. CoAX's
``coax_human_model_and_baselines.csv`` only carries hard 0/1 predictions for
its DT/KNN/MLP baselines, not the probabilities NLL needs, and no baseline
predictions exist yet for CoXAM -- both are reported with a null
``fit_table`` and a ``note`` explaining the gap, rather than a fabricated
number computed from a hard prediction.

Run ``fit_sim2real_baselines_to_participants.py`` first.
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

HUMAN_DATA = REPO_ROOT / "assets" / "human_data"
OUTPUT = REPO_ROOT / "assets" / "human_vs_model_fit_stats.json"

SIM2REAL_CONDITIONS = ("faithful", "sparse", "robust", "sparse_robust")
CONDITION_LABELS = {
    "faithful": "Faithful",
    "sparse": "Sparse",
    "robust": "Robust",
    "sparse_robust": "Sparse + robust",
}
BASELINE_LABELS = {
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}


def _cell(nll: pd.Series, bic: pd.Series) -> dict[str, Any]:
    return {
        "nll_mean": float(nll.mean()),
        "nll_sd": float(nll.std(ddof=1)) if len(nll) > 1 else 0.0,
        "bic_mean": float(bic.mean()),
        "bic_sd": float(bic.std(ddof=1)) if len(bic) > 1 else 0.0,
        "n": int(len(nll)),
    }


def build_sim2real() -> Optional[dict[str, Any]]:
    fits_path = HUMAN_DATA / "Sim2Real" / "sim2real_participant_fits.csv"
    baselines_path = HUMAN_DATA / "Sim2Real" / "sim2real_baseline_fits.csv"
    if not fits_path.is_file() or not baselines_path.is_file():
        return None

    own = pd.read_csv(fits_path)
    baselines = pd.read_csv(baselines_path)
    facets = [c for c in SIM2REAL_CONDITIONS if c in set(own["exp_property"]) | set(baselines["exp_property"])]

    models = [
        {
            "name": "sim2real",
            "label": "sim2real",
            "is_target": True,
            "cells": {
                CONDITION_LABELS[facet]: _cell(
                    rows["test_nll_model_participant"], rows["test_bic_model_participant"]
                )
                for facet, rows in own.groupby("exp_property")
                if facet in facets
            },
        }
    ]
    for baseline, rows in baselines.groupby("baseline"):
        models.append(
            {
                "name": baseline,
                "label": BASELINE_LABELS.get(baseline, baseline),
                "is_target": False,
                "cells": {
                    CONDITION_LABELS[facet]: _cell(
                        facet_rows["test_nll_model_participant"],
                        facet_rows["test_bic_model_participant"],
                    )
                    for facet, facet_rows in rows.groupby("exp_property")
                    if facet in facets
                },
            }
        )

    return {
        "name": "sim2real",
        "facets": [CONDITION_LABELS[f] for f in facets],
        "models": models,
        "note": (
            "sim2real is the study's fitted cognitive model (best strategy per "
            "participant); the three baselines are DecisionTreeBaseline, "
            "LogisticRegressionBaseline and MLPBaseline from "
            "src/cognitive_models/baseline_models, trained once per "
            "exp_property on the ten training cases and scored against every "
            "participant's real answers. Lower is better."
        ),
    }


def build_coax_gap() -> dict[str, Any]:
    return {
        "name": "coax",
        "facets": [],
        "models": [],
        "note": (
            "assets/human_data/CoAX/coax_human_model_and_baselines.csv carries only "
            "hard 0/1 predictions for the DT/KNN/MLP baselines, not the "
            "probabilities NLL needs -- no fit table until those baselines are "
            "re-scored with predict_proba."
        ),
    }


def build_coxam_gap() -> dict[str, Any]:
    return {
        "name": "coxam",
        "facets": [],
        "models": [],
        "note": (
            "No DT/LR baseline predictions exist yet under "
            "assets/human_data/CoXAM -- no fit table until one is fitted."
        ),
    }


def main() -> int:
    sim2real = build_sim2real()
    studies = []
    if sim2real is not None:
        studies.append(sim2real)
    else:
        print(
            "skipping sim2real: run "
            "fit_sim2real_baselines_to_participants.py and assets/build_human_data.py first"
        )
    studies.append(build_coax_gap())
    studies.append(build_coxam_gap())

    payload = {
        "generated_from": "assets/build_human_vs_model_fit_stats.py",
        "bic_formula": "n_parameters * log(n_trials) + 2 * total_nll",
        "studies": studies,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    for study in studies:
        models = study.get("models") or []
        print(f"{study['name']:9s} {len(models)} model(s), facets={study.get('facets')}")
        for model in models:
            for facet, cell in model["cells"].items():
                print(
                    f"    {model['name']:22s} {facet:16s} "
                    f"nll={cell['nll_mean']:.3f}+/-{cell['nll_sd']:.3f} "
                    f"bic={cell['bic_mean']:.1f}+/-{cell['bic_sd']:.1f} n={cell['n']}"
                )
    print(f"\nwrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

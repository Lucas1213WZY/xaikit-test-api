"""Build the NLL/BIC model-fit comparison tables for the UI to render.

Emits one JSON (``assets/human_vs_model_fit_stats.json``). Each study carries
a list of ``tables`` (usually one; CoXAM has two -- forward and
counterfactual are different tasks with different baseline models, not one
table with more rows), and each table has facets, one row per model (the
study's own cognitive model plus any machine-proxy baselines), and one
NLL/BIC cell per facet.

sim2real's table is computed live: ``fit_sim2real_baselines_to_participants.py``
fits DT/logistic-regression/MLP proxies and scores them against real
participant answers, and this script reads that output.

CoAX and CoXAM's tables are **hardcoded from the published paper tables**,
not computed here. ``coax_human_model_and_baselines.csv`` only carries hard
0/1 predictions for its DT/KNN/MLP baselines, not the probabilities NLL
needs, and no baseline predictions exist yet for CoXAM under
``assets/human_data/CoXAM`` -- the same arrangement ``FITTED_SIM2REAL_PARAMS``
and ``FITTED_COAX_PARAMS`` use for values that cannot be reproduced from
what is checked into this repo. A cell absent from a model's ``cells`` dict
is the paper table's "---": that baseline was not fit for that condition
(e.g. Decision Tree is not evaluated at Weights, since there is no decision
tree to compare against a Weights-only explanation).

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
        "tables": [
            {
                "title": "Counterfactual comparison, by explanation property",
                "facets": [CONDITION_LABELS[f] for f in facets],
                "models": models,
                "note": (
                    "sim2real is the study's fitted cognitive model (best strategy "
                    "per participant); the three baselines are DecisionTreeBaseline, "
                    "LogisticRegressionBaseline and MLPBaseline from "
                    "src/cognitive_models/baseline_models, trained once per "
                    "exp_property on the ten training cases and scored against every "
                    "participant's real answers. Lower is better."
                ),
            }
        ],
    }


# -- CoAX (hardcoded from the published table) -----------------------------

#: cells[model][facet] = (nll_mean, nll_sd, bic_mean, bic_sd). Transcribed from
#: the paper's Table (proxy-model-fits): mean +/- SD across participants,
#: averaged over all three corpus datasets (adult, forest_cover, wine_quality).
_COAX_FACETS = ["None XAI", "Importance XAI", "Attribution XAI"]
_COAX_CELLS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "decision_tree": {
        "None XAI": (0.61, 0.10, 27.7, 3.6),
        "Importance XAI": (0.60, 0.10, 27.4, 3.6),
        "Attribution XAI": (0.58, 0.09, 26.7, 3.2),
    },
    "knn": {
        "None XAI": (0.58, 0.14, 26.7, 5.0),
        "Importance XAI": (0.59, 0.11, 27.0, 4.0),
        "Attribution XAI": (0.46, 0.16, 22.3, 5.8),
    },
    "mlp": {
        "None XAI": (0.61, 0.07, 27.7, 2.5),
        "Importance XAI": (0.60, 0.09, 27.4, 3.2),
        "Attribution XAI": (0.43, 0.11, 21.3, 4.0),
    },
    "coax": {
        "None XAI": (0.47, 0.22, 26.0, 7.9),
        "Importance XAI": (0.38, 0.23, 25.8, 7.2),
        "Attribution XAI": (0.31, 0.21, 25.7, 7.6),
    },
}
_COAX_LABELS = {"decision_tree": "DT", "knn": "KNN", "mlp": "MLP", "coax": "CoAX"}


def _paper_cell(nll_mean: float, nll_sd: float, bic_mean: float, bic_sd: float) -> dict[str, Any]:
    # n is not reported in the paper table, unlike sim2real's live-computed
    # cells -- present as null rather than omitted, so callers can rely on the
    # key always existing.
    return {"nll_mean": nll_mean, "nll_sd": nll_sd, "bic_mean": bic_mean, "bic_sd": bic_sd, "n": None}


def build_coax() -> dict[str, Any]:
    models = [
        {
            "name": name,
            "label": _COAX_LABELS[name],
            "is_target": name == "coax",
            "cells": {facet: _paper_cell(*values) for facet, values in facets.items()},
        }
        for name, facets in _COAX_CELLS.items()
    ]
    return {
        "name": "coax",
        "tables": [
            {
                "title": "Forward simulation, by XAI type",
                "facets": _COAX_FACETS,
                "models": models,
                "note": (
                    "From the published paper table, not computed in this repo -- "
                    "assets/human_data/CoAX/coax_human_model_and_baselines.csv carries "
                    "only hard 0/1 predictions for the DT/KNN/MLP baselines, not the "
                    "probabilities NLL needs. Averaged over all participants from the "
                    "3 corpus datasets (adult, forest_cover, wine_quality). Lower is "
                    "better."
                ),
            }
        ],
    }


# -- CoXAM (hardcoded from the published tables) ---------------------------

_COXAM_FACETS = [
    "Wine Quality — Rules", "Wine Quality — Weights", "Wine Quality — Hybrid",
    "Mushrooms — Rules", "Mushrooms — Weights", "Mushrooms — Hybrid",
]

#: cells[model][facet] = (nll, bic). Transcribed from the paper's forward-task
#: table (coxam-vs-baselines-forward). A facet absent from a model is the
#: paper's "---": Decision Tree is not evaluated where the shown explanation
#: is Weights-only, Linear Regression not where it is Rules-only -- each
#: baseline is only comparable against the explanation family it models.
_COXAM_FORWARD_CELLS: dict[str, dict[str, tuple[float, float]]] = {
    "decision_tree": {
        "Wine Quality — Rules": (26.8, 57.3),
        "Wine Quality — Hybrid": (26.2, 56.1),
        "Mushrooms — Rules": (21.7, 47.1),
        "Mushrooms — Hybrid": (29.7, 63.1),
    },
    "linear_regression": {
        "Wine Quality — Weights": (28.0, 59.7),
        "Wine Quality — Hybrid": (26.8, 57.3),
        "Mushrooms — Weights": (37.0, 77.7),
        "Mushrooms — Hybrid": (35.5, 74.7),
    },
    "knn_no_xai": {
        "Wine Quality — Rules": (29.7, 66.8),
        "Wine Quality — Weights": (30.1, 67.6),
        "Wine Quality — Hybrid": (29.9, 67.2),
        "Mushrooms — Rules": (24.0, 55.4),
        "Mushrooms — Weights": (28.1, 63.6),
        "Mushrooms — Hybrid": (27.2, 61.8),
    },
    "coxam": {
        "Wine Quality — Rules": (18.9, 48.8),
        "Wine Quality — Weights": (19.9, 50.9),
        "Wine Quality — Hybrid": (20.2, 51.5),
        "Mushrooms — Rules": (20.7, 52.5),
        "Mushrooms — Weights": (21.5, 54.1),
        "Mushrooms — Hybrid": (20.8, 52.7),
    },
}
_COXAM_FORWARD_LABELS = {
    "decision_tree": "Decision Tree",
    "linear_regression": "Linear Regression",
    "knn_no_xai": "KNN w/o XAI",
    "coxam": "CoXAM",
}

#: Transcribed from the paper's counterfactual-task table (coxam-vs-baselines).
#: All six facets are present for every model -- unlike the forward table,
#: none of these baselines are explanation-family-specific.
_COXAM_COUNTERFACTUAL_CELLS: dict[str, dict[str, tuple[float, float]]] = {
    "random": {facet: (71.67, 143.34) for facet in _COXAM_FACETS},
    "global_shap": {
        "Wine Quality — Rules": (42.1, 84.2),
        "Wine Quality — Weights": (56.3, 112.6),
        "Wine Quality — Hybrid": (54.7, 106.4),
        "Mushrooms — Rules": (50.2, 100.4),
        "Mushrooms — Weights": (55.1, 110.2),
        "Mushrooms — Hybrid": (51.0, 102.0),
    },
    "coxam": {
        "Wine Quality — Rules": (35.6, 71.2),
        "Wine Quality — Weights": (51.8, 103.6),
        "Wine Quality — Hybrid": (45.0, 90.0),
        "Mushrooms — Rules": (45.8, 91.6),
        "Mushrooms — Weights": (50.6, 101.2),
        "Mushrooms — Hybrid": (46.6, 93.2),
    },
}
_COXAM_COUNTERFACTUAL_LABELS = {"random": "Random", "global_shap": "Global SHAP", "coxam": "CoXAM"}


def _paper_cell_no_sd(nll: float, bic: float) -> dict[str, Any]:
    # Neither the forward nor the counterfactual CoXAM table reports a
    # per-cell SD (single value, not mean +/- SD like CoAX's) -- null rather
    # than a fabricated 0.0, so a reader cannot mistake it for "measured, no
    # spread".
    return {"nll_mean": nll, "nll_sd": None, "bic_mean": bic, "bic_sd": None, "n": None}


def _coxam_models(
    cells: dict[str, dict[str, tuple[float, float]]],
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": labels[name],
            "is_target": name == "coxam",
            "cells": {facet: _paper_cell_no_sd(*values) for facet, values in facets.items()},
        }
        for name, facets in cells.items()
    ]


def build_coxam() -> dict[str, Any]:
    return {
        "name": "coxam",
        "tables": [
            {
                "title": "Forward simulation, by dataset and explanation family",
                "facets": _COXAM_FACETS,
                "models": _coxam_models(_COXAM_FORWARD_CELLS, _COXAM_FORWARD_LABELS),
                "note": (
                    "From the published paper table, not computed in this repo -- no "
                    "DT/LR baseline predictions exist yet under assets/human_data/CoXAM. "
                    "A model absent for a facet is the paper's '---': that baseline is "
                    "only comparable against the explanation family it models (Decision "
                    "Tree vs Rules, Linear Regression vs Weights; both still apply at "
                    "Hybrid). Lower is better."
                ),
            },
            {
                "title": "Counterfactual, by dataset and explanation family",
                "facets": _COXAM_FACETS,
                "models": _coxam_models(_COXAM_COUNTERFACTUAL_CELLS, _COXAM_COUNTERFACTUAL_LABELS),
                "note": (
                    "From the published paper table, not computed in this repo. Random "
                    "is constant by construction (no XAI-dependence); Global SHAP is a "
                    "single global attribution baseline, not fit per participant. Lower "
                    "is better."
                ),
            },
        ],
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
    studies.append(build_coax())
    studies.append(build_coxam())

    payload = {
        "generated_from": "assets/build_human_vs_model_fit_stats.py",
        "bic_formula": "n_parameters * log(n_trials) + 2 * total_nll",
        "studies": studies,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    for study in studies:
        tables = study.get("tables") or []
        print(f"{study['name']:9s} {len(tables)} table(s)")
        for table in tables:
            print(f"  {table['title']}  facets={table['facets']}")
            for model in table["models"]:
                for facet, cell in model["cells"].items():
                    nll_sd = "" if cell["nll_sd"] is None else f"+/-{cell['nll_sd']:.3f}"
                    bic_sd = "" if cell["bic_sd"] is None else f"+/-{cell['bic_sd']:.1f}"
                    print(
                        f"    {model['name']:20s} {facet:24s} "
                        f"nll={cell['nll_mean']:.3f}{nll_sd} "
                        f"bic={cell['bic_mean']:.1f}{bic_sd} n={cell['n']}"
                    )
    print(f"\nwrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

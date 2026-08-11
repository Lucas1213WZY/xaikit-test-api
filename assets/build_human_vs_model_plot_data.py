"""Build the human-vs-cognitive-model comparison data for all three studies.

Emits one JSON the interactive report embeds. Every bar is a mean over
*participant* means, not over trials, so a participant with more trials does not
count for more; the error bar is a 95% confidence interval on those means.

Each study asks its participants a different question, so each panel names its
own dependent variable rather than pretending one scale spans all three:

* CoAX and CoXAM forward -- "did you predict what the AI would say?", scored as
  agreement with the AI's prediction.
* CoXAM counterfactual -- "did your edit flip the AI's prediction?"
* sim2real -- "is the changed case higher or lower?", scored against the answer
  key.

No participant identifier reaches the output: everything here is an aggregate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Every source below is an anonymised table under assets/human_data. Nothing
# here reads server_runs/, which keys participants on their raw session code.
HUMAN_DATA = REPO_ROOT / "assets" / "human_data"
OUTPUT = REPO_ROOT / "assets" / "human_vs_model_plot_data.json"

SIM2REAL_CONDITIONS = ("faithful", "sparse", "robust", "sparse_robust")

#: The datasets broken out into per-XAI-type panels, nested with the
#: "shown"/"not shown" split like build_coxam()'s dataset x shown layout.
#: Mushrooms is deliberately excluded from this per-condition grid (see the
#: note where these panels are built); the by-dataset overview panel below
#: still covers all four.
COAX_FIGURE_DATASETS = ("adult", "covertype", "wine quality")

#: The study tables spell the same dataset several ways.
DATASET_LABELS = {
    "wine quality": "Wine quality",
    "wine_quality": "Wine quality",
    "Wine Quality": "Wine quality",
    "Mushrooms": "Mushrooms",
    "forest_cover": "Forest cover",
    "covertype": "Forest cover",
    "adult": "Adult income",
    "mushrooms": "Mushrooms",
}

#: The published figure relabels the two explanation families, so the report uses
#: the same words: decision tree -> Rules, logistic regression -> Weights. The
#: replay export spells the conditions out in full; the older 50-participant fit
#: used DT/LR/DT+LR, so both vocabularies are mapped.
CONDITION_FAMILY_LABELS = {
    "Decision Tree": "Rules",
    "Linear Regression": "Weights",
    "Hybrid": "Hybrid",
    "DT": "Rules",
    "LR": "Weights",
    "DT+LR": "Hybrid",
}

CONDITION_LABELS = {
    "DT": "Decision tree",
    "LR": "Logistic regression",
    "DT+LR": "Hybrid",
    "Hybrid": "Hybrid",
    "dt": "Decision tree",
    "lr": "Logistic regression",
    "faithful": "Faithful",
    "sparse": "Sparse",
    "robust": "Robust",
    "sparse_robust": "Sparse + robust",
}


def _label(value: Any, table: dict[str, str]) -> str:
    return table.get(str(value), str(value))


def _participant_summary(
    frame: pd.DataFrame,
    *,
    participant: str,
    group: str,
    value: str,
) -> list[dict[str, Any]]:
    """Mean over participant means, with a 95% CI half-width on those means.

    Delegates to ``src.result_visualizer`` so the API and this script cannot
    drift onto two different estimators.
    """
    from src.result_visualizer import participant_summary

    summary = participant_summary(
        frame, participant_column=participant, group_column=group, value_column=value
    )
    return [
        {"group": str(row["group"]), "mean": row["mean"], "error": row["ci95"], "n": int(row["n"])}
        for _, row in summary.iterrows()
    ]


def _panel(
    frame: pd.DataFrame,
    *,
    participant: str,
    group: str,
    series: dict[str, str],
    group_labels: dict[str, str],
    title: str,
    dv: str,
    note: str,
    order: Optional[list[str]] = None,
) -> dict[str, Any]:
    """One grouped-bar panel: a category axis, one bar per agent."""
    summaries = {
        name: {row["group"]: row for row in _participant_summary(
            frame[frame[column].notna()], participant=participant, group=group, value=column
        )}
        for name, column in series.items()
    }
    keys = order or sorted({key for summary in summaries.values() for key in summary})
    return {
        "title": title,
        "dv": dv,
        "note": note,
        "categories": [_label(key, group_labels) for key in keys],
        "series": [
            {
                "name": name,
                "values": [summaries[name].get(key, {}).get("mean") for key in keys],
                "error": [summaries[name].get(key, {}).get("error") for key in keys],
                "n": [summaries[name].get(key, {}).get("n") for key in keys],
            }
            for name in series
        ],
    }


# -- CoAX -----------------------------------------------------------------


def build_coax() -> dict[str, Any]:
    """Human vs CoAX vs the three ML baselines, on test trials only.

    The published table scores the model agents on training trials too, but the
    humans only ever appear on test trials -- comparing the two as shipped would
    hand the models a different trial set, so everything is cut to ``Test``.
    """
    frame = pd.read_csv(HUMAN_DATA / "CoAX" / "coax_human_model_and_baselines.csv", low_memory=False)
    frame = frame[frame["trialType"] == "Test"].copy()
    frame["Correct"] = pd.to_numeric(frame["Correct"], errors="coerce")

    # Model agents carry no participant id; each is one deterministic agent, so
    # its own rows are the unit and a synthetic single id keeps the estimator
    # identical for humans (mean over people) and models (mean over trials).
    frame["unit"] = np.where(
        frame["agent"] == "Human", frame["Participant ID"].astype(str), frame["agent"].astype(str)
    )

    panels = []
    # Human vs CoAX only, matching the study's own UI ("CoAX vs. Human
    # Prevalence and Accuracy" in CoAX/simulation_mockup/xai_chart.html, whose
    # Agent field carries just those two). The DT/KNN/MLP rows in the published
    # table are ML baselines, not cognitive models, so they are left out.
    agents = ("Human", "CoAX")
    # The DV is the published table's own ``Correct`` -- forward simulation
    # accuracy. It cannot be re-derived here: ``predicted`` is empty for both
    # agents on test trials and ``AI Prediction`` is empty for CoAX, so the
    # column is taken exactly as the study computed it.
    dv = "Forward simulation accuracy"
    note = (
        "Test trials only. The published table carries no participant id for CoAX, so only "
        "the human bars carry a confidence interval — the CoAX bar is one deterministic agent "
        "(n = 1), not a sample."
    )
    xai_order = ["none", "importance", "attribution"]
    xai_labels = {"none": "None", "importance": "Importance", "attribution": "Attribution"}

    def _bars(subset: pd.DataFrame, group: str, keys: list[Any], labels: dict[Any, str]) -> dict[str, Any]:
        summaries = {
            agent: {
                row["group"]: row
                for row in _participant_summary(
                    subset[subset["agent"] == agent], participant="unit", group=group, value="Correct"
                )
            }
            for agent in agents
        }
        return {
            "categories": [_label(key, labels) for key in keys],
            "series": [
                {
                    "name": agent,
                    "values": [summaries[agent].get(key, {}).get("mean") for key in keys],
                    "error": [summaries[agent].get(key, {}).get("error") for key in keys],
                    "n": [summaries[agent].get(key, {}).get("n") for key in keys],
                }
                for agent in agents
            ],
        }

    # Laid out like the published figure: one panel per dataset x
    # explanation-shown cell, with the three XAI types on the x axis --
    # nested/multifaceted the same way build_coxam() lays out its dataset x
    # shown panels. Mushrooms is excluded here (kept only in the pooled
    # "Overall — by dataset" panel below): its own difficulty (0.56) sits far
    # enough from the other three that mixing it into this per-condition grid
    # was called out explicitly as unwanted.
    for data_id in COAX_FIGURE_DATASETS:
        for shown in ("w/o XAI", "w/ XAI"):
            subset = frame[
                (frame["Tested w/ XAI"].astype(str) == shown) & (frame["dataId"] == data_id)
            ]
            if subset.empty:
                continue
            keys = [key for key in xai_order if key in set(subset["XAIType"].astype(str))]
            panels.append(
                {
                    "title": (
                        f"Forward simulation — "
                        f"{DATASET_LABELS.get(data_id, data_id).lower()}, {shown}"
                    ),
                    "dv": dv,
                    "note": note,
                    **_bars(subset, "XAIType", keys, xai_labels),
                }
            )

    # Overall view, every dataset side by side. The published figure covers one
    # dataset; the table covers four, and this is where the other three appear.
    keys = sorted(set(frame["dataId"].astype(str)))
    panels.insert(
        0,
        {"title": "Overall — by dataset", "dv": dv,
         "note": note + " Pooled over explanation type.",
         **_bars(frame, "dataId", keys, DATASET_LABELS)},
    )

    return {
        "name": "CoAX",
        "task": "Forward simulation — predict the AI's answer",
        "participants": int(frame.loc[frame["agent"] == "Human", "Participant ID"].nunique()),
        "panels": panels,
    }


# -- CoXAM ----------------------------------------------------------------


def build_coxam() -> dict[str, Any]:
    """CoXAM's own published fits: forward on mushrooms, counterfactual on both."""
    forward = pd.read_csv(HUMAN_DATA / "CoXAM" / "coxam_forward_fit_mushrooms.csv", low_memory=False)
    # The full replay (270 participants, both datasets) rather than the
    # 50-participant rl_fit_trials subset it supersedes.
    counterfactual = pd.read_csv(
        HUMAN_DATA / "CoXAM" / "coxam_counterfactual_replay.csv", low_memory=False
    )

    # Laid out like the published forward figure: one panel per dataset x
    # explanation-shown cell, with the explanation family on the x axis and
    # Human beside CoXAM. Complexity is deliberately not a panel -- the figure
    # does not split on it. Only mushrooms can be drawn: the wine-quality
    # forward fit exported parameters, never per-trial predictions, so the
    # figure's Wine Quality panels have no source in assets/.
    condition_order = ["Rules", "Weights", "Hybrid"]
    forward = forward.copy()
    forward["family"] = forward["Condition"].map(CONDITION_FAMILY_LABELS).fillna(forward["Condition"])
    panels = []
    for shown, label in ((False, "w/o XAI"), (True, "w/ XAI")):
        subset = forward[forward["Tested w/ XAI"].astype(bool) == shown]
        if subset.empty:
            continue
        panels.append(
            _panel(
                subset,
                participant="Participant Id",
                group="family",
                series={"Human": "Response==AI", "CoXAM": "Model==AI"},
                group_labels={},
                title=f"Forward simulation — mushrooms, {label}",
                dv="Forward accuracy",
                note=(
                    "Rules = decision tree, Weights = logistic regression. Mushrooms only; "
                    "the wine-quality forward fit exported parameters, not per-trial predictions."
                ),
                order=[c for c in condition_order if c in set(subset["family"])],
            )
        )

    counterfactual = counterfactual.copy()
    counterfactual["Changed AI prediction"] = pd.to_numeric(
        counterfactual["Changed AI prediction"], errors="coerce"
    )
    counterfactual["Model changed AI prediction"] = pd.to_numeric(
        counterfactual["Model changed AI prediction"], errors="coerce"
    )
    # Laid out like the published counterfactual figure: one panel per dataset x
    # explanation-shown cell, family on the x axis. The figure's companion plot
    # scored the same edits against the *explainer* rather than the AI; that is a
    # different measure and is deliberately not mixed in here.
    shown = counterfactual["Tested w/ XAI"].astype(str).str.strip()
    unexpected = set(shown.unique()) - {"w/ XAI", "w/o XAI"}
    if unexpected:
        raise ValueError(f"Unexpected 'Tested w/ XAI' value(s): {sorted(unexpected)}")
    counterfactual["shown"] = shown
    counterfactual["family"] = (
        counterfactual["condition"].map(CONDITION_FAMILY_LABELS).fillna(counterfactual["condition"])
    )

    # Overall view first: both datasets side by side, pooled over condition and
    # whether the explanation was shown.
    panels.append(
        _panel(
            counterfactual,
            participant="Participant Id",
            group="dataId",
            series={"Human": "Changed AI prediction", "CoXAM": "Model changed AI prediction"},
            group_labels=DATASET_LABELS,
            title="Overall — counterfactual, by dataset",
            dv="Counterfactual accuracy",
            note=(
                "Both datasets side by side, pooled over condition and whether the "
                "explanation was shown."
            ),
            order=["Wine Quality", "Mushrooms"],
        )
    )
    # Wine quality first, matching the figure's panel order a-d.
    for data_id in ("Wine Quality", "Mushrooms"):
        for shown_label in ("w/o XAI", "w/ XAI"):
            subset = counterfactual[
                (counterfactual["dataId"] == data_id) & (counterfactual["shown"] == shown_label)
            ]
            if subset.empty:
                continue
            panels.append(
                _panel(
                    subset,
                    participant="Participant Id",
                    group="family",
                    series={
                        "Human": "Changed AI prediction",
                        "CoXAM": "Model changed AI prediction",
                    },
                    group_labels={},
                    title=(
                        f"Counterfactual — {DATASET_LABELS.get(data_id, data_id).lower()}, "
                        f"{shown_label}"
                    ),
                    dv="Counterfactual accuracy",
                    note=(
                        "Whether the edit flipped the AI's prediction — a different question "
                        "from the forward panels. Rules = decision tree, Weights = logistic "
                        "regression."
                    ),
                    order=[c for c in condition_order if c in set(subset["family"])],
                )
            )

    return {
        "name": "CoXAM",
        "task": (
            f"Forward simulation ({forward['Participant Id'].nunique()} participants, mushrooms) "
            f"and counterfactual editing ({counterfactual['Participant Id'].nunique()}, both datasets)"
        ),
        "participants": int(
            pd.concat([forward["Participant Id"], counterfactual["Participant Id"]]).nunique()
        ),
        "panels": panels,
    }


# -- sim2real -------------------------------------------------------------


def build_sim2real() -> dict[str, Any]:
    """Human vs the fitted strategy, per explanation property."""
    # The anonymised consolidated copy, not server_runs/: the fitter keyed
    # participants on the raw session code, so reading its output directly would
    # put un-anonymised identifiers behind the plots.
    path = HUMAN_DATA / "Sim2Real" / "sim2real_fit_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run assets/build_human_data.py to consolidate the sim2real fit."
        )
    frame = pd.read_csv(path)
    frame["condition"] = frame["exp_property"]
    frame["human_correct"] = (frame["human_label"] == frame["correct_label"]).astype(float)
    frame["model_correct"] = (frame["model_label"] == frame["correct_label"]).astype(float)
    frame["model_matches_human"] = (frame["model_label"] == frame["human_label"]).astype(float)

    order = list(SIM2REAL_CONDITIONS)
    panels = [
        _panel(
            frame,
            participant="participant_id",
            group="condition",
            series={"Human": "human_correct", "sim2real": "model_correct"},
            group_labels=CONDITION_LABELS,
            title="Accuracy, by explanation property",
            dv="Correct on the counterfactual question",
            note="Scored against the answer key.",
            order=order,
        ),
        _panel(
            frame,
            participant="participant_id",
            group="condition",
            series={"sim2real": "model_matches_human"},
            group_labels=CONDITION_LABELS,
            title="How often the model reproduces the person",
            dv="Model answer matches the participant's",
            note="This is what the fit optimises — not accuracy.",
            order=order,
        ),
    ]

    return {
        "name": "sim2real",
        "task": "Counterfactual comparison — is the changed case higher or lower?",
        "participants": int(frame["participant_id"].nunique()),
        "panels": panels,
    }


def main() -> int:
    payload = {
        "generated_from": "assets/build_human_vs_model_plot_data.py",
        "estimator": (
            "mean of participant means; error bars are 95% confidence intervals on those "
            "means, using the Student-t multiplier for each group's own n"
        ),
        "studies": [build_coax(), build_coxam(), build_sim2real()],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    for study in payload["studies"]:
        print(f"{study['name']:9s} {study['participants']:4d} participants, {len(study['panels'])} panels")
        for panel in study["panels"]:
            values = ", ".join(
                f"{s['name']}={'/'.join('--' if v is None else f'{v:.3f}' for v in s['values'])}"
                for s in panel["series"]
            )
            print(f"    {panel['title']}: {panel['categories']} {values}")
    print(f"\nwrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

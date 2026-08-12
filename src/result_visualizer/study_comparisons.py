"""One-call human-vs-cognitive-model comparisons for the bundled studies.

``comparison_panel`` builds a panel from any frame; this module knows where each
study's fitted results live under ``assets/human_data`` and what its panels
should be, so a caller does not have to::

    from src.result_visualizer import human_vs_model_report

    human_vs_model_report()                 # every study that has data
    human_vs_model_report("coxam")          # just one
    panels = study_comparison("coxam")      # the ComparisonStudy, to inspect

Every source is an anonymised table -- nothing here reads ``server_runs/``,
which keys participants on their raw session code. A study with no data on disk
is skipped rather than raising, so the call works on a fresh checkout where
``assets/human_data`` has not been built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .human_vs_model import ComparisonPanel, ComparisonStudy, comparison_panel, render_comparison_report

__all__ = [
    "STUDY_NAMES",
    "available_studies",
    "human_vs_model_report",
    "load_coxam_fitted_trials",
    "load_sim2real_fitted_trials",
    "study_comparison",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN_DATA = REPO_ROOT / "assets" / "human_data"

#: The studies this module knows how to build, in report order.
STUDY_NAMES: tuple[str, ...] = ("coax", "coxam", "sim2real")

#: Explanation families, in the two vocabularies the tables use.
_FAMILY_LABELS = {
    "Decision Tree": "Rules", "Linear Regression": "Weights", "Hybrid": "Hybrid",
    "DT": "Rules", "LR": "Weights", "DT+LR": "Hybrid",
}
_FAMILY_ORDER = ["Rules", "Weights", "Hybrid"]

_DATASET_LABELS = {
    "adult": "Adult income", "covertype": "Forest cover", "forest_cover": "Forest cover",
    "mushrooms": "Mushrooms", "Mushrooms": "Mushrooms",
    "wine quality": "Wine quality", "wine_quality": "Wine quality", "Wine Quality": "Wine quality",
}

_SIM2REAL_LABELS = {
    "faithful": "Faithful", "sparse": "Sparse",
    "robust": "Robust", "sparse_robust": "Sparse + robust",
}
_SIM2REAL_ORDER = ["faithful", "sparse", "robust", "sparse_robust"]

#: The dataset the published CoAX figure covers. Pooling all four would move the
#: bars with the dataset mix rather than with the explanation type.
_COAX_FIGURE_DATASET = "adult"


def _read(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path, low_memory=False) if path.is_file() else None


def load_coxam_fitted_trials(task: str = "forward") -> Optional[pd.DataFrame]:
    """CoXAM's published per-trial fit for one task, or None if absent.

    Unlike CoAX there is no replay to run: CoXAM's fit was produced by the
    study's own notebooks and shipped per trial, already carrying both the
    human response and the fitted model's, so these tables *are* the replay.

    Args:
        task: ``forward`` (mushrooms, 137 participants) or ``counterfactual``
            (both datasets, 270).

    Raises:
        ValueError: If ``task`` is neither of those.
    """
    files = {
        "forward": "coxam_forward_fit_mushrooms.csv",
        "counterfactual": "coxam_counterfactual_replay.csv",
    }
    if task not in files:
        raise ValueError(f"task must be 'forward' or 'counterfactual', not {task!r}.")
    return _read(HUMAN_DATA / "CoXAM" / files[task])


def load_sim2real_fitted_trials() -> Optional[pd.DataFrame]:
    """sim2real's per-trial fitted predictions, renumbered onto anonymised ids.

    The fitter writes these keyed on the raw ``clinical_<code>`` session code;
    ``assets/build_human_data.py`` maps them onto the ``1..N`` participants, and
    this reads that consolidated copy rather than ``server_runs/``.
    """
    return _read(HUMAN_DATA / "Sim2Real" / "sim2real_fit_predictions.csv")


# -- CoAX -----------------------------------------------------------------


def _coax(replay: Optional[pd.DataFrame] = None) -> Optional[ComparisonStudy]:
    """Replay every logged CoAX trial with that participant's own fitted strategy.

    Runs ``run_coax_human_replay`` rather than reading the published comparison
    table. The published table sets ``Participant ID`` to NaN for the model, so
    CoAX would be one deterministic agent with no spread; the replay carries the
    participant through, so both series get real confidence intervals.

    Args:
        replay: A replay already computed, to avoid re-running it.
    """
    if replay is None:
        try:
            from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
                run_coax_human_replay,
            )

            replay = run_coax_human_replay(on_missing_fit="skip")
        except (FileNotFoundError, RuntimeError):
            return None

    rows = replay[replay["human_comparable"]].copy()
    rows = rows[rows["human_correct"].notna() & rows["cognitive_correct_vs_ai"].notna()]
    if rows.empty:
        return None

    rows["human_matches_ai"] = rows["human_correct"].astype(float)
    rows["model_matches_ai"] = rows["cognitive_correct_vs_ai"].astype(float)
    rows["shown"] = rows["tested_w_xai"].map({True: "w/ XAI", False: "w/o XAI"})

    series = {"Human": "human_matches_ai", "CoAX": "model_matches_ai"}
    note = (
        "Each participant replayed with their own fitted CoAX strategy, scored on the one "
        "step the person actually answered."
    )
    panels = [
        comparison_panel(
            rows, participant_column="participantId", group_column="dataId", series=series,
            title="Overall — by dataset", dv="Forward Simulation Accuracy",
            note=note + " Pooled over explanation type.", group_labels=_DATASET_LABELS,
        )
    ]
    for shown in ("w/o XAI", "w/ XAI"):
        subset = rows[(rows["shown"] == shown) & (rows["dataId"] == _COAX_FIGURE_DATASET)]
        if subset.empty:
            continue
        panels.append(
            comparison_panel(
                subset, participant_column="participantId", group_column="xai_type", series=series,
                title=f"Forward simulation — adult income, {shown}",
                dv="Forward Simulation Accuracy", note=note,
                order=[k for k in ("none", "importance", "attribution") if k in set(subset["xai_type"])],
                group_labels={"none": "None", "importance": "Importance", "attribution": "Attribution"},
            )
        )
    return ComparisonStudy(
        name="CoAX", task="Forward simulation — predict the AI's answer",
        participants=int(rows["participantId"].nunique()), panels=panels,
    )


# -- CoXAM ----------------------------------------------------------------


def _coxam() -> Optional[ComparisonStudy]:
    forward = load_coxam_fitted_trials("forward")
    replay = load_coxam_fitted_trials("counterfactual")
    if forward is None and replay is None:
        return None

    panels: list[ComparisonPanel] = []
    participants: set = set()

    if forward is not None:
        forward = forward.copy()
        forward["family"] = forward["Condition"].map(_FAMILY_LABELS).fillna(forward["Condition"])
        participants |= set(forward["Participant Id"].dropna())
        for shown in (False, True):
            subset = forward[forward["Tested w/ XAI"].astype(bool) == shown]
            if subset.empty:
                continue
            panels.append(comparison_panel(
                subset, participant_column="Participant Id", group_column="family",
                series={"Human": "Response==AI", "CoXAM": "Model==AI"},
                title=f"Forward simulation — mushrooms, {'w/ XAI' if shown else 'w/o XAI'}",
                dv="Forward Simulation Accuracy",
                note=("Rules = decision tree, Weights = logistic regression. Mushrooms only; the "
                      "wine-quality forward fit exported parameters, not per-trial predictions."),
                order=[f for f in _FAMILY_ORDER if f in set(subset["family"])],
            ))

    if replay is not None:
        replay = replay.copy()
        replay["family"] = replay["condition"].map(_FAMILY_LABELS).fillna(replay["condition"])
        for column in ("Changed AI prediction", "Model changed AI prediction"):
            replay[column] = pd.to_numeric(replay[column], errors="coerce")
        participants |= set(replay["Participant Id"].dropna())
        series = {"Human": "Changed AI prediction", "CoXAM": "Model changed AI prediction"}
        note = ("Whether the edit flipped the AI's prediction — a different question from the "
                "forward panels. Rules = decision tree, Weights = logistic regression.")
        panels.append(comparison_panel(
            replay, participant_column="Participant Id", group_column="dataId", series=series,
            title="Overall — counterfactual, by dataset", dv="Counterfactual Simulation Accuracy",
            note="Both datasets, pooled over condition and whether the explanation was shown.",
            order=["Wine Quality", "Mushrooms"], group_labels=_DATASET_LABELS,
        ))
        for data_id in ("Wine Quality", "Mushrooms"):
            for shown in ("w/o XAI", "w/ XAI"):
                subset = replay[
                    (replay["dataId"] == data_id) & (replay["Tested w/ XAI"].astype(str).str.strip() == shown)
                ]
                if subset.empty:
                    continue
                panels.append(comparison_panel(
                    subset, participant_column="Participant Id", group_column="family", series=series,
                    title=f"Counterfactual — {_DATASET_LABELS.get(data_id, data_id).lower()}, {shown}",
                    dv="Counterfactual Simulation Accuracy", note=note,
                    order=[f for f in _FAMILY_ORDER if f in set(subset["family"])],
                ))

    tasks = []
    if forward is not None:
        tasks.append(f"forward simulation ({forward['Participant Id'].nunique()}, mushrooms)")
    if replay is not None:
        tasks.append(f"counterfactual editing ({replay['Participant Id'].nunique()}, both datasets)")
    return ComparisonStudy(
        name="CoXAM", task=" and ".join(tasks).capitalize(),
        participants=len(participants), panels=panels,
    )


# -- sim2real -------------------------------------------------------------


def _sim2real() -> Optional[ComparisonStudy]:
    frame = load_sim2real_fitted_trials()
    if frame is None:
        return None

    frame = frame.copy()
    frame["human_correct"] = (frame["human_label"] == frame["correct_label"]).astype(float)
    frame["model_correct"] = (frame["model_label"] == frame["correct_label"]).astype(float)
    frame["model_matches_human"] = (frame["model_label"] == frame["human_label"]).astype(float)
    order = [c for c in _SIM2REAL_ORDER if c in set(frame["exp_property"])]

    return ComparisonStudy(
        name="sim2real",
        task="Counterfactual comparison — is the changed case higher or lower?",
        participants=int(frame["participant_id"].nunique()),
        panels=[
            comparison_panel(
                frame, participant_column="participant_id", group_column="exp_property",
                series={"Human": "human_correct", "sim2real": "model_correct"},
                title="Accuracy, by explanation property",
                dv="Counterfactual Simulation Accuracy",
                note="Scored against the answer key.", order=order, group_labels=_SIM2REAL_LABELS,
            ),
            comparison_panel(
                frame, participant_column="participant_id", group_column="exp_property",
                series={"sim2real": "model_matches_human"},
                title="How often the model reproduces the human responses",
                dv="Rate of Matches",
                note="This is what the fit optimises — not accuracy.",
                order=order, group_labels=_SIM2REAL_LABELS,
            ),
        ],
    )


_BUILDERS = {"coax": _coax, "coxam": _coxam, "sim2real": _sim2real}


def study_comparison(name: str) -> Optional[ComparisonStudy]:
    """The human-vs-model panels for one study, or None if its data is absent.

    Args:
        name: ``coax``, ``coxam`` or ``sim2real`` (case-insensitive).

    Raises:
        ValueError: If ``name`` is not one of :data:`STUDY_NAMES`.
    """
    key = str(name).lower().strip()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown study {name!r}. Use one of: {', '.join(STUDY_NAMES)}.")
    return _BUILDERS[key]()


def available_studies() -> list[str]:
    """Which studies have fitted results on disk right now."""
    return [name for name in STUDY_NAMES if study_comparison(name) is not None]


def human_vs_model_report(
    studies: Optional[str | Iterable[str]] = None,
    output_path: Optional[Path | str] = None,
    *,
    title: str = "Human vs cognitive model",
) -> Path:
    """Build and render every available human-vs-model comparison in one call.

    Args:
        studies: A study name, several names, or None for every one that has
            data.
        output_path: Where to write the HTML. Defaults to
            ``assets/human_vs_model_report.html``, or a per-study name when a
            single study is requested.
        title: Page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If none of the requested studies has data on disk.
    """
    if studies is None:
        names: Sequence[str] = STUDY_NAMES
    elif isinstance(studies, str):
        names = [studies]
    else:
        names = list(studies)

    built = [study for study in (study_comparison(name) for name in names) if study is not None]
    if not built:
        raise FileNotFoundError(
            f"No fitted results found under {HUMAN_DATA} for {list(names)}. "
            "Run assets/build_human_data.py to consolidate them."
        )

    if output_path is None:
        stem = "human_vs_model_report" if len(built) > 1 else f"human_vs_model_{built[0].name.lower()}"
        output_path = REPO_ROOT / "assets" / f"{stem}.html"
    return render_comparison_report(built, output_path, title=title)

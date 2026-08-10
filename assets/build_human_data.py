"""Consolidate every study's human data into ``assets/human_data``, anonymised.

Source files are read but never written. Each framework's participants are
renumbered ``1..N`` and the raw identifiers are dropped from the output, so the
consolidated tables carry no Prolific ID, study ID or session ID.

Numbering is per framework -- CoAX participant 1 is not CoXAM participant 1 --
and is assigned by sorting the raw identifiers lexicographically, which keeps
the mapping deterministic without leaking the order people took part in.

``participant_id_map.csv`` is the re-identification key. It is the one output
that must never be shared; ``assets/human_data/`` is gitignored in full, which
covers it, but treat it as you would the raw data.

Usage::

    python assets/build_human_data.py            # write the consolidated tables
    python assets/build_human_data.py --check    # verify they are up to date
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HUMAN_DATA = REPO_ROOT / "assets" / "human_data"

# Run as a script, sys.path[0] is assets/, so the fitter's screening rule --
# imported lazily in load_sim2real_trials -- would not resolve.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COAX_ROOT = REPO_ROOT / "src" / "cognitive_models" / "CoAX"
COXAM_ROOT = REPO_ROOT / "src" / "cognitive_models" / "CoXAM"

#: sim2real ships one file per participant, outside the repository.
SIM2REAL_DIR = Path(
    "/Users/wangzhuoyulucas/Library/Mobile Documents/com~apple~CloudDocs"
    "/XAI Ubicomp Lab/XAI API/data_sim2real/data/uci_data"
)

#: ``clinical_<code>.csv`` -- one file per *sitting*, not per person. The code
#: matches the ``code`` column inside the file, but two people sat the study
#: twice under different codes, so ``prolific_id`` is the participant key.
SIM2REAL_FILE_PATTERN = re.compile(r"^clinical_(?P<code>\d+)\.csv$")

#: Dropped from the sim2real output. ``prolific_id`` becomes ``participant_id``
#: and ``code`` becomes ``participant_session``; the rest identify the Prolific
#: session directly.
SIM2REAL_IDENTIFIER_COLUMNS = ("code", "prolific_id", "study_id", "session_id")

#: CoXAM's published participant fits, produced by the study's own notebooks and
#: kept outside the repository alongside the raw data. These are the fits the
#: paper reports -- a DDM-with-timing model fitted jointly on response NLL and
#: time MAE -- not the RL meta-policy parameters ``fit_coxam_to_participants``
#: searches. The two are different model families and are not interchangeable.
COXAM_FIT_DIR = Path(
    "/Users/wangzhuoyulucas/Library/Mobile Documents/com~apple~CloudDocs"
    "/XAI Ubicomp Lab/CoXAM/xaikit-test-global-exp-RLdev"
)

#: ``appId`` and ``dataId`` are the same thing -- the dataset's name. ``dataId``
#: is the public spelling, so the consolidated tables use it throughout. The
#: CoXAM fit exports spell it three more ways again.
COLUMN_RENAMES = {
    "appId": "dataId",
    "AppId": "dataId",
    "App Id": "dataId",
    "app_id": "dataId",
}


@dataclass(frozen=True)
class Table:
    """One source table and where its anonymised copy is written."""

    source: Path
    output: str
    id_column: str
    note: str = ""
    #: Drop rows whose every non-identifier column is empty. The JMP exports
    #: carry one such row, with a column *name* sitting in the identifier field,
    #: which would otherwise fail the anonymisation as an unmapped participant.
    drop_blank_rows: bool = False


@dataclass
class Framework:
    """A study, its tables, and the participant numbering shared across them."""

    name: str
    directory: str
    tables: Sequence[Table] = field(default_factory=tuple)


FRAMEWORKS: tuple[Framework, ...] = (
    Framework(
        name="CoAX",
        directory="CoAX",
        tables=(
            Table(
                source=COAX_ROOT / "data" / "user study results" / "3-datasets-jan-09-2026-trials.csv",
                output="coax_human_trials.csv",
                id_column="Participant ID",
                note="one row per trial step, all 4 datasets",
            ),
            Table(
                source=COAX_ROOT / "results" / "pop_em_subset" / "refit_all_assignments_detlocal-Jan-10.csv",
                output="coax_fitted_strategies.csv",
                id_column="Participant ID",
                note="fitted strategy + parameters per participant-condition",
            ),
            Table(
                source=COAX_ROOT / "results" / "Human and CoAX and Baselines.csv",
                output="coax_human_model_and_baselines.csv",
                id_column="Participant ID",
                note="published human vs CoAX vs baseline comparison",
            ),
            Table(
                source=COAX_ROOT / "results" / "04-12-2025" / "trial_data_no_explanation_individual_16-54.csv",
                output="coax_individual_2025-12-04_no_explanation.csv",
                id_column="participant_id",
            ),
            Table(
                source=COAX_ROOT / "results" / "04-12-2025" / "trial_data_w_explanation_individual_16-54.csv",
                output="coax_individual_2025-12-04_with_explanation.csv",
                id_column="participant_id",
            ),
            Table(
                source=COAX_ROOT / "results" / "05-12-2025" / "trial_data_no_explanation_individual_11-09.csv",
                output="coax_individual_2025-12-05_no_explanation.csv",
                id_column="participant_id",
            ),
            Table(
                source=COAX_ROOT / "results" / "05-12-2025" / "trial_data_w_explanation_individual_11-09.csv",
                output="coax_individual_2025-12-05_with_explanation.csv",
                id_column="participant_id",
            ),
        ),
    ),
    Framework(
        name="CoXAM",
        directory="CoXAM",
        tables=(
            Table(
                source=COXAM_ROOT / "datasets" / "mushrooms and wine quality user data v0.1.csv",
                output="coxam_human_trials.csv",
                id_column="Participant Id",
                note="one row per trial, mushrooms + wine quality",
            ),
            Table(
                source=COXAM_FIT_DIR / "exports" / "participant_trial_dump.csv",
                output="coxam_forward_fit_mushrooms.csv",
                id_column="Participant Id",
                note="per-trial forward fit: fitted parameters, p_resp and NLL",
            ),
            Table(
                source=COXAM_FIT_DIR / "shared_params_trial_dump.csv",
                output="coxam_forward_fit_mushrooms_shared_params.csv",
                id_column="Participant Id",
                note="the same trials refitted with parameters shared across participants",
            ),
            Table(
                source=COXAM_FIT_DIR / "participant_parameters_fit_dt_v0.1.csv",
                output="coxam_forward_params_wine_quality_dt.csv",
                id_column="Participant Id",
                note="fitted parameters only -- no per-trial predictions",
            ),
            Table(
                source=COXAM_FIT_DIR / "participant_parameters_fit_lr_v0.1.csv",
                output="coxam_forward_params_wine_quality_lr.csv",
                id_column="Participant Id",
                note="fitted parameters only; calculation and heuristic strategies",
            ),
            Table(
                source=COXAM_FIT_DIR / "outputs" / "rl_fit_trials.csv",
                output="coxam_counterfactual_fit.csv",
                id_column="Participant Id",
                note="per-trial counterfactual fit, 50-participant subset of the replay below",
            ),
            Table(
                source=HUMAN_DATA / "CoXAM" / "overall_counterfactual_resultsv0.2.xlsx",
                output="coxam_counterfactual_replay.csv",
                id_column="Participant Id",
                note="the full counterfactual replay: 270 participants, both datasets",
                drop_blank_rows=True,
            ),
        ),
    ),
    Framework(name="Sim2Real", directory="Sim2Real"),
)

MAP_FILE = "participant_id_map.csv"


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _load_table(table: Table) -> pd.DataFrame:
    """Read a source table and apply its declared cleanup."""
    frame = _read(table.source)
    if table.id_column not in frame.columns:
        raise KeyError(f"{table.source} has no {table.id_column!r} column.")
    if table.drop_blank_rows:
        others = frame.drop(columns=[table.id_column])
        frame = frame[~others.isna().all(axis=1)].reset_index(drop=True)
    return frame


def _raw_ids(table: Table) -> list[str]:
    return _load_table(table)[table.id_column].dropna().astype(str).tolist()


def build_participant_map(raw_ids: Iterable[str]) -> dict[str, int]:
    """Number the unique identifiers ``1..N`` in lexicographic order."""
    return {raw: index for index, raw in enumerate(sorted(set(raw_ids)), start=1)}


def _anonymise(frame: pd.DataFrame, id_column: str, mapping: dict[str, int]) -> pd.DataFrame:
    """Replace the identifier column in place, keeping its name and position."""
    out = frame.copy()
    ids = out[id_column].astype(str).map(mapping)
    unmapped = out[id_column].notna() & ids.isna()
    if unmapped.any():
        missing = sorted(out.loc[unmapped, id_column].astype(str).unique())[:3]
        raise RuntimeError(f"{len(missing)} identifier(s) outside the map, e.g. {missing}.")
    out[id_column] = ids.astype("Int64")
    return out.rename(columns=COLUMN_RENAMES)


def _sim2real_prolific_id(frame: pd.DataFrame, name: str) -> str:
    """The one Prolific ID in a sitting's file."""
    if "prolific_id" not in frame.columns:
        raise RuntimeError(f"{name} has no prolific_id column.")
    values = frame["prolific_id"].dropna().astype(str).str.strip()
    unique = sorted(set(values[values != ""]))
    if len(unique) != 1:
        raise RuntimeError(f"{name} holds {len(unique)} Prolific IDs, expected exactly 1.")
    return unique[0]


def load_sim2real_trials() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Combine the eligible sim2real sittings into one anonymised table.

    Participants are keyed on ``prolific_id``, not on the ``clinical_<code>``
    file name: two people sat the study twice under different codes, so the
    files count sittings rather than people. ``participant_session`` numbers a
    person's sittings in code order.

    Sittings are screened with the fitter's own rule -- exactly one response for
    every expected test qid -- by calling
    :func:`load_screened_participant_trials`, so this table and the fits agree
    on who counts by construction rather than by a re-implemented rule.

    The files come in two schema versions -- 46 add ``strategy``, ``confidence``
    and ``question_type`` -- so the union of columns is used and the others
    carry blanks there rather than being silently truncated.
    """
    if not SIM2REAL_DIR.is_dir():
        raise FileNotFoundError(f"sim2real source directory not found: {SIM2REAL_DIR}")

    from src.cognitive_models.cognitive_models.fit_sim2real_attribution_sum_to_participants import (
        load_screened_participant_trials,
    )

    _, screening = load_screened_participant_trials(SIM2REAL_DIR)
    eligible_files = set(screening.loc[screening["eligible"], "source_file"])

    paths = sorted(
        (path for path in SIM2REAL_DIR.glob("*.csv") if SIM2REAL_FILE_PATTERN.match(path.name)),
        key=lambda path: path.name,
    )
    if not paths:
        raise RuntimeError(f"No clinical_<code>.csv files under {SIM2REAL_DIR}.")

    # Read every sitting so screened-out ones still resolve their Prolific ID,
    # which is what tells us whether an excluded sitting belongs to someone who
    # also has an eligible one.
    sittings: list[tuple[str, str, pd.DataFrame]] = []
    for path in paths:
        frame = _read(path)
        code = SIM2REAL_FILE_PATTERN.match(path.name)["code"]
        inside = frame["code"].dropna().astype(str).unique() if "code" in frame else []
        if len(inside) and set(inside) != {code}:
            raise RuntimeError(f"{path.name} holds code(s) {sorted(inside)}, not {code}.")
        sittings.append((code, _sim2real_prolific_id(frame, path.name), frame))

    kept = [(code, raw, frame) for code, raw, frame in sittings if f"clinical_{code}.csv" in eligible_files]
    if not kept:
        raise RuntimeError("No sim2real sitting passed screening.")

    mapping = build_participant_map(raw for _, raw, _ in kept)
    sessions: dict[str, int] = {}
    frames = []
    map_rows: list[dict[str, object]] = []
    for code, raw, frame in kept:
        sessions[raw] = sessions.get(raw, 0) + 1
        frame = frame.copy()
        frame.insert(0, "participant_session", sessions[raw])
        frame.insert(0, "participant_id", mapping[raw])
        frames.append(frame.drop(columns=list(SIM2REAL_IDENTIFIER_COLUMNS), errors="ignore"))
        map_rows.append(
            {
                "framework": "Sim2Real",
                "raw_participant_id": raw,
                "participant_id": mapping[raw],
                "raw_session_code": code,
                "participant_session": sessions[raw],
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    lead = ["participant_id", "participant_session"]
    combined = combined[[*lead, *(c for c in combined.columns if c not in lead)]]

    # Report screening without naming anyone: the raw code identifies a sitting.
    audit = screening.drop(columns=["participant_id", "source_file"], errors="ignore").copy()
    audit.insert(
        0,
        "participant_id",
        pd.Series(
            [
                mapping.get(raw) if f"clinical_{code}.csv" in eligible_files else None
                for code, raw, _ in sittings
            ],
            dtype="Int64",
        ),
    )
    return combined, audit.sort_values("participant_id", na_position="last"), map_rows


#: The per-condition sim2real fit outputs, keyed on the raw session code.
SIM2REAL_FIT_DIR = REPO_ROOT / "server_runs" / "sim2real_refit_strategies"
SIM2REAL_EXP_PROPERTIES = ("faithful", "sparse", "robust", "sparse_robust")
SIM2REAL_FIT_FILES = {
    "participant_predictions.csv": "sim2real_fit_predictions.csv",
    "participant_fits.csv": "sim2real_participant_fits.csv",
}


def load_sim2real_fit_outputs(
    code_to_participant: Mapping[str, int],
    *,
    fit_dir: Path = SIM2REAL_FIT_DIR,
) -> list[tuple[str, pd.DataFrame]]:
    """The fitted sim2real outputs, renumbered onto the anonymised participants.

    The fitter keyed participants on the raw ``clinical_<code>`` session code, so
    these files carry codes rather than the ``1..N`` numbering. Without this the
    plots would read identifiers straight out of ``server_runs/``, which is both
    un-anonymised and outside ``assets``.
    """
    if not fit_dir.is_dir():
        return []

    outputs: list[tuple[str, pd.DataFrame]] = []
    for source_name, output_name in SIM2REAL_FIT_FILES.items():
        frames = []
        for exp_property in SIM2REAL_EXP_PROPERTIES:
            path = fit_dir / exp_property / source_name
            if not path.is_file():
                continue
            chunk = pd.read_csv(path)
            if "exp_property" in chunk.columns:
                present = set(chunk["exp_property"].dropna().astype(str).unique())
                if present - {exp_property}:
                    raise RuntimeError(
                        f"{path} is filed under {exp_property!r} but holds {sorted(present)}."
                    )
            else:
                chunk.insert(0, "exp_property", exp_property)
            codes = chunk["participant_id"].astype(str)
            unmapped = sorted(set(codes) - set(code_to_participant))
            if unmapped:
                raise RuntimeError(
                    f"{path} holds session code(s) outside the map, e.g. {unmapped[:3]}."
                )
            chunk["participant_id"] = codes.map(code_to_participant).astype("Int64")
            frames.append(chunk)
        if frames:
            outputs.append((output_name, pd.concat(frames, ignore_index=True, sort=False)))
    return outputs


def build(*, check: bool = False) -> int:
    """Write (or verify) every consolidated table. Returns the exit status."""
    HUMAN_DATA.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    map_rows: list[dict[str, object]] = []

    for framework in FRAMEWORKS:
        target = HUMAN_DATA / framework.directory
        target.mkdir(parents=True, exist_ok=True)

        if framework.name == "Sim2Real":
            combined, audit, sim2real_rows = load_sim2real_trials()
            outputs = [
                ("sim2real_human_trials.csv", combined),
                ("sim2real_screening.csv", audit),
            ]
            outputs.extend(
                load_sim2real_fit_outputs(
                    {str(row["raw_session_code"]): int(row["participant_id"]) for row in sim2real_rows}
                )
            )
            participants = {row["raw_participant_id"]: row["participant_id"] for row in sim2real_rows}
        else:
            missing = [t.source for t in framework.tables if not t.source.is_file()]
            if missing:
                raise FileNotFoundError(f"{framework.name} source missing: {missing[0]}")
            participants = build_participant_map(
                raw for table in framework.tables for raw in _raw_ids(table)
            )
            outputs = [
                (table.output, _anonymise(_load_table(table), table.id_column, participants))
                for table in framework.tables
            ]

        for name, frame in outputs:
            path = target / name
            written = frame.to_csv(index=False)
            if check:
                if not path.is_file() or path.read_text() != written:
                    stale.append(str(path.relative_to(REPO_ROOT)))
            else:
                path.write_text(written)
            print(f"  {framework.name:9s} {name:52s} {len(frame):7,d} rows")

        if framework.name == "Sim2Real":
            map_rows.extend(sorted(sim2real_rows, key=lambda row: (row["participant_id"], row["participant_session"])))
        else:
            map_rows.extend(
                {"framework": framework.name, "raw_participant_id": raw, "participant_id": new}
                for raw, new in sorted(participants.items(), key=lambda item: item[1])
            )
        print(f"  {framework.name:9s} {'participants':52s} {len(participants):7,d}")

    map_frame = pd.DataFrame(map_rows)
    map_path = HUMAN_DATA / MAP_FILE
    map_text = map_frame.to_csv(index=False)
    if check:
        if not map_path.is_file() or map_path.read_text() != map_text:
            stale.append(str(map_path.relative_to(REPO_ROOT)))
    else:
        map_path.write_text(map_text)

    if check and stale:
        print("\nOut of date:")
        for path in stale:
            print(f"  {path}")
        print("\nRun: python assets/build_human_data.py")
        return 1

    print(f"\n{len(map_frame):,} participants mapped across {len(FRAMEWORKS)} studies.")
    print("Up to date." if check else f"Key written to {map_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the consolidated tables match their sources; write nothing",
    )
    return build(check=parser.parse_args().check)


if __name__ == "__main__":
    sys.exit(main())

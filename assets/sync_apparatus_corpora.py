"""Sync the repository's corpora with the apparatus that served the user study.

``xaikit-test-ui-apparatus`` is what the real participants saw, so it is the
authority on instance values, AI predictions and explanation vectors. This
script copies it in, in two layers:

* ``assets/apparatus/`` keeps each source file **verbatim**, under the
  apparatus's own ``appId`` / ``v{i}`` spelling, so the import can always be
  re-verified against its source.
* the canonical asset paths get the same data **normalized** to this repo's
  schema (``dataId`` / ``x{i}``), which is what every loader already reads.

Two rules keep the sync from silently changing what a runner sees.

*The canonical files keep their own row selection.* ``assets/explanations/CoAX``
deliberately carries three datasets with one XAI method each, not the
apparatus's four-by-two; syncing replaces those rows' **values** and never adds
or drops a key. CoXAM's canonical tables are trimmed to the six datasets the
apparatus carries, as requested.

*Nothing outside the apparatus is invented.* A canonical row whose key the
apparatus does not carry is reported and left alone rather than dropped.

Why this matters: ``assets/explanations/CoAX/attribution.csv`` was found to
differ from both the apparatus and the CoAX study's own ``standard set`` on
every row -- same magnitudes, ~1-2% scatter, ratios centred on 1.0, which is
LIME/SHAP re-run under a different seed. Simulations reading it were therefore
not seeing the vectors the participants saw.

Run with ``--check`` to report differences without writing anything.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"
DEFAULT_APPARATUS = REPO_ROOT.parent / "xaikit-test-ui-apparatus"

#: Verbatim copies land here, mirroring the apparatus's own layout.
VERBATIM_ROOT = ASSETS / "apparatus"

#: ``(apparatus file, canonical asset file, key columns)``.
#:
#: ``none.csv`` is the AI-prediction record and appears twice on purpose: the
#: CoAX repository reads it from ``explanations/``, the dataset loaders from
#: ``ai_dataset/``.
SYNC_PLAN: list[tuple[str, str, list[str]]] = [
    # -- CoAX: three datasets, one XAI method each; keys are preserved --------
    ("local/xai_methods/values.csv", "ai_dataset/coax/values.csv", ["dataId", "instanceId"]),
    ("local/xai_methods/none.csv", "ai_dataset/coax/none.csv", ["dataId", "instanceId", "modelName"]),
    ("local/xai_methods/none.csv", "explanations/CoAX/none.csv", ["dataId", "instanceId", "modelName"]),
    ("local/xai_methods/attribution.csv", "explanations/CoAX/attribution.csv",
     ["dataId", "instanceId", "expMethod"]),
    ("local/xai_methods/importance.csv", "explanations/CoAX/importance.csv",
     ["dataId", "instanceId", "expMethod"]),
    # -- CoXAM: trimmed to the six datasets the apparatus serves --------------
    ("global/xai_methods/values.csv", "ai_dataset/coxam/values.csv", ["dataId", "instanceId"]),
    ("global/xai_methods/none.csv", "ai_dataset/coxam/none.csv", ["dataId", "instanceId", "modelName"]),
    ("global/xai_methods/none.csv", "explanations/CoXAM/none.csv", ["dataId", "instanceId", "modelName"]),
    ("global/xai_methods/decision_tree.csv", "explanations/CoXAM/decision_tree.csv",
     ["dataId", "model", "depth"]),
    ("global/xai_methods/logistic_regression.csv", "explanations/CoXAM/logistic_regression.csv",
     ["dataId", "model", "variant"]),
]

#: Canonical tables rebuilt from the apparatus's full row set rather than having
#: their existing keys preserved -- the request was for CoXAM to carry exactly
#: the apparatus's six datasets, so its tables are replaced outright.
REBUILD_WHOLE: frozenset[str] = frozenset(
    {
        "ai_dataset/coxam/values.csv",
        "ai_dataset/coxam/none.csv",
        "explanations/CoXAM/none.csv",
        "explanations/CoXAM/decision_tree.csv",
        "explanations/CoXAM/logistic_regression.csv",
    }
)

_FEATURE_PREFIXES = ("v", "x")


def to_repo_schema(table: pd.DataFrame) -> pd.DataFrame:
    """Apparatus spelling (``appId``, ``v{i}``) -> repo spelling (``dataId``, ``x{i}``).

    Only feature columns are renamed: ``v3``, ``v3_min``, ``v3_options`` become
    ``x3``, ``x3_min``, ``x3_options``, while ``variant`` and ``value`` keep
    their names because the character after the ``v`` is not a digit.
    """
    renamed = {"appId": "dataId"}
    for column in table.columns:
        if column.startswith("v") and column[1:2].isdigit():
            renamed[column] = "x" + column[1:]
    return table.rename(columns=renamed)


def _as_keys(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = table.copy()
    for key in keys:
        out[key] = out[key].astype(str)
    return out


def sync_one(
    apparatus_root: Path, source_rel: str, target_rel: str, keys: list[str], *, check: bool
) -> dict[str, object]:
    """Sync one canonical table; returns a report row."""
    source_path = apparatus_root / source_rel
    target_path = ASSETS / target_rel
    source = to_repo_schema(pd.read_csv(source_path, low_memory=False))

    if not target_path.exists():
        raise FileNotFoundError(target_path)
    target = pd.read_csv(target_path, low_memory=False)

    usable_keys = [key for key in keys if key in source.columns and key in target.columns]
    shared_columns = [column for column in target.columns if column in source.columns]

    if target_rel in REBUILD_WHOLE:
        rebuilt = source[[c for c in source.columns if c in target.columns or True]].copy()
        rebuilt = rebuilt[[c for c in source.columns]]
        changed_cells = None
        missing_keys = 0
        note = f"rebuilt {len(target)} -> {len(rebuilt)} rows"
    else:
        source_keyed = _as_keys(source, usable_keys).set_index(usable_keys)
        target_keyed = _as_keys(target, usable_keys).set_index(usable_keys)
        present = target_keyed.index.isin(source_keyed.index)
        missing_keys = int((~present).sum())

        rebuilt = target_keyed.copy()
        value_columns = [c for c in shared_columns if c not in usable_keys]
        aligned = source_keyed.loc[target_keyed.index[present], value_columns]
        before = rebuilt.loc[present, value_columns].copy()
        rebuilt.loc[present, value_columns] = aligned.to_numpy()

        changed_cells = 0
        for column in value_columns:
            left, right = before[column], rebuilt.loc[present, column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                differs = ~np.isclose(
                    left.fillna(-9e12).to_numpy(float), right.fillna(-9e12).to_numpy(float), atol=1e-12
                )
            else:
                differs = left.astype(str).to_numpy() != right.astype(str).to_numpy()
            changed_cells += int(differs.sum())
        rebuilt = rebuilt.reset_index()
        rebuilt = rebuilt[list(target.columns)]
        note = f"{changed_cells} cell(s) updated"

    if not check:
        backup = target_path.with_suffix(target_path.suffix + ".pre-apparatus.bak")
        if not backup.exists():
            shutil.copy2(target_path, backup)
        rebuilt.to_csv(target_path, index=False)

    return {
        "target": target_rel,
        "source": source_rel,
        "rows_before": len(target),
        "rows_after": len(rebuilt),
        "keys_not_in_apparatus": missing_keys,
        "note": note,
    }


#: The XAI method each dataset was shown with in the CoAX user study. The
#: canonical CoAX assets deliberately carry one method per dataset, and this is
#: the assignment the study itself used -- so extending them for the study never
#: introduces a second method.
#:
#: ``mushrooms`` is deliberately absent. The trial log carries it, but the
#: published refit does not (see ``coax_human_replay.FITTED_DATA_IDS``), so it
#: is not a CoAX dataset for our purposes and its 92 instances are not imported.
COAX_STUDY_METHOD: dict[str, str] = {
    "adult": "lime",
    "forest_cover": "shap",
    "wine_quality": "lime",
}

#: Where the human trial log lives, for :func:`extend_coax_for_human_study`.
COAX_HUMAN_TRIALS = (
    REPO_ROOT
    / "src" / "cognitive_models" / "CoAX" / "data" / "user study results"
    / "3-datasets-jan-09-2026-trials.csv"
)


def extend_coax_for_human_study(apparatus_root: Path, *, check: bool) -> list[dict[str, object]]:
    """Add whatever the CoAX assets still lack to cover the human study.

    Replaying a participant needs every instance that participant saw. The
    canonical assets are a curated subset, so they were missing all of
    ``mushrooms`` and two ``wine_quality`` instances. This adds exactly the
    missing rows -- never a second XAI method for a dataset, and never a row the
    apparatus does not carry -- leaving existing rows untouched.
    """
    log = pd.read_csv(COAX_HUMAN_TRIALS, low_memory=False)
    log = log[log["Step"].astype(str).str.strip().str.lower() == "infer"]
    needed = {
        str(data_id): set(int(value) for value in rows["Instance Index"].dropna())
        for data_id, rows in log.groupby("appId")
        if str(data_id) in COAX_STUDY_METHOD
    }

    targets = [
        ("local/xai_methods/values.csv", "ai_dataset/coax/values.csv", False),
        ("local/xai_methods/none.csv", "ai_dataset/coax/none.csv", False),
        ("local/xai_methods/attribution.csv", "explanations/CoAX/attribution.csv", True),
        ("local/xai_methods/importance.csv", "explanations/CoAX/importance.csv", True),
    ]

    report: list[dict[str, object]] = []
    for source_rel, target_rel, by_method in targets:
        source = to_repo_schema(pd.read_csv(apparatus_root / source_rel, low_memory=False))
        target_path = ASSETS / target_rel
        target = pd.read_csv(target_path, low_memory=False)

        additions = []
        for data_id, instance_ids in needed.items():
            have = target[target["dataId"].astype(str) == data_id]
            method = COAX_STUDY_METHOD[data_id]
            if by_method and not have.empty:
                have = have[have["expMethod"].astype(str) == method]
            missing = sorted(instance_ids - set(have["instanceId"].astype(int)))
            if not missing:
                continue
            rows = source[
                (source["dataId"].astype(str) == data_id)
                & (source["instanceId"].astype(int).isin(missing))
            ]
            if by_method:
                rows = rows[rows["expMethod"].astype(str) == method]
            additions.append(rows[[c for c in target.columns if c in rows.columns]])

        added = 0 if not additions else int(sum(len(frame) for frame in additions))
        if additions and not check:
            combined = pd.concat([target, *additions], ignore_index=True, sort=False)
            combined = combined.sort_values(
                [c for c in ("dataId", "expMethod", "instanceId") if c in combined.columns],
                kind="stable",
            ).reset_index(drop=True)
            combined.to_csv(target_path, index=False)
        report.append(
            {"target": target_rel, "rows_before": len(target), "rows_added": added}
        )
    return report


def copy_verbatim(apparatus_root: Path, *, check: bool) -> list[str]:
    """Keep each apparatus source file under its own schema, unmodified."""
    copied = []
    for source_rel in sorted({source for source, _target, _keys in SYNC_PLAN}):
        destination = VERBATIM_ROOT / source_rel
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(apparatus_root / source_rel, destination)
        copied.append(str(destination.relative_to(ASSETS)))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apparatus-root", type=Path, default=DEFAULT_APPARATUS)
    parser.add_argument(
        "--check", action="store_true", help="Report what would change without writing."
    )
    args = parser.parse_args()

    if not args.apparatus_root.is_dir():
        raise SystemExit(f"Apparatus repository not found at {args.apparatus_root}")

    verbatim = copy_verbatim(args.apparatus_root, check=args.check)
    report = [
        sync_one(args.apparatus_root, source, target, keys, check=args.check)
        for source, target, keys in SYNC_PLAN
    ]

    extended = extend_coax_for_human_study(args.apparatus_root, check=args.check)

    print(("CHECK ONLY -- nothing written" if args.check else "Synced") + f" from {args.apparatus_root}")
    print(f"\nverbatim copies ({len(verbatim)}):")
    for path in verbatim:
        print(f"   assets/{path}")
    print()
    print(pd.DataFrame(report).to_string(index=False))
    print("\nCoAX assets extended to cover every instance the human study used:")
    print(pd.DataFrame(extended).to_string(index=False))


if __name__ == "__main__":
    main()

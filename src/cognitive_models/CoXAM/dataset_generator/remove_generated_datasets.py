from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CSV_NAMES = [
    "metadata.csv",
    "values.csv",
    "none.csv",
    "logistic_regression.csv",
    "decision_tree.csv",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def backup_csv(path: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / f"{path.stem}_{timestamp}{path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove generated appId rows from dataset CSVs.")
    parser.add_argument("app_ids", nargs="+", help="appId values to remove.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backups before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = project_root() / "datasets"
    app_ids = set(args.app_ids)

    for csv_name in CSV_NAMES:
        path = data_dir / csv_name
        df = pd.read_csv(path)
        if "appId" not in df.columns:
            print(f"{csv_name}: skipped, no appId column", flush=True)
            continue
        mask = df["appId"].isin(app_ids)
        removed = int(mask.sum())
        print(f"{csv_name}: removing {removed} rows", flush=True)
        if args.dry_run or removed == 0:
            continue
        if not args.no_backup:
            backup_csv(path)
        df.loc[~mask].to_csv(path, index=False)

    if args.dry_run:
        print("Dry run only; no files written.", flush=True)


if __name__ == "__main__":
    main()


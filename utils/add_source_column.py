#!/usr/bin/env python3
"""
Append/fill a final source column for maternal and newborn CSV exports.

Default value: "DHIS"
Facility override value: "EMR"
Default folders:
  - utils/imnid/maternal
  - utils/imnid/newborn
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


DHIS_SOURCE_VALUE = "DHIS"
EMR_SOURCE_VALUE = "EMR"
EMR_FACILITY_UIDS = {
    "QSD7RkcT8v2",  # Axum Referral Hospital, Tigray
    "Q6ryvdOY3Tj",  # Tula Primary Hospital, Sidama
    "WNzF8NcSHLu",  # Olenchity Primary Hospital
}
EMR_FACILITY_NAMES = {
    "axum referral hospital",
    "axum referal hospital",
    "axum referal",
    "axum referral",
    "tula primary hospital",
    "olenchity primary hospital",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def row_source_value(row: dict[str, str], default_source_value: str) -> str:
    org_unit_uid = str(row.get("orgUnit") or row.get("orgUnit_uid") or "").strip()
    org_unit_name = normalize_text(row.get("orgUnit_name") or row.get("facility") or "")

    if (
        org_unit_uid in EMR_FACILITY_UIDS
        or org_unit_name in EMR_FACILITY_NAMES
    ):
        return EMR_SOURCE_VALUE

    return default_source_value


def assign_source_column(df: pd.DataFrame, default_source_value: str = DHIS_SOURCE_VALUE) -> pd.DataFrame:
    """Return a copy with final source column set from facility UID/name rules."""
    if df is None or df.empty:
        return df

    result = df.copy()
    rows = result.to_dict(orient="records")
    result["source"] = [
        row_source_value(row, default_source_value) for row in rows
    ]

    columns = [col for col in result.columns if col != "source"] + ["source"]
    return result.reindex(columns=columns)


def update_csv_source_column(path: Path, source_value: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            return

        original_fields = list(reader.fieldnames)
        fields_without_source = [field for field in original_fields if field != "source"]
        output_fields = fields_without_source + ["source"]

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=str(path.parent),
            suffix=".tmp",
        ) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()

            for row in reader:
                row["source"] = row_source_value(row, source_value)
                writer.writerow(row)

            temp_path = Path(outfile.name)

    temp_path.replace(path)


def update_folder(folder: Path, source_value: str) -> int:
    if not folder.exists():
        print(f"Skipped missing folder: {folder}")
        return 0

    updated = 0
    for csv_path in sorted(folder.glob("*.csv")):
        update_csv_source_column(csv_path, source_value)
        updated += 1
        print(f"Updated: {csv_path}")

    return updated


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_base_dir = script_dir / "imnid"

    parser = argparse.ArgumentParser(
        description="Append/fill a final source column in maternal and newborn CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base_dir,
        help="Base directory containing maternal and newborn folders.",
    )
    parser.add_argument(
        "--source-value",
        default=DHIS_SOURCE_VALUE,
        help='Default value to write into the source column. Default: "DHIS".',
    )
    args = parser.parse_args()

    folders = [args.base_dir / "maternal", args.base_dir / "newborn"]
    total_updated = sum(update_folder(folder, args.source_value) for folder in folders)
    print(f"Done. Updated {total_updated} CSV file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

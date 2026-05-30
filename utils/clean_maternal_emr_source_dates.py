#!/usr/bin/env python3
"""
Remove out-of-window EMR-only maternal rows from CSV exports.

Only rows where source == "EMR" are eligible for removal.
Rows are kept when enrollment_date is from 2025-04-01 through today.
DHIS and EMR to DHIS rows are never removed by this script.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile


START_DATE = date(2025, 4, 1)


def normalize_source(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def ec_to_gc(raw_value: str) -> date | None:
    raw_date = raw_value.strip().split("T")[0].split(" ")[0]
    parts = raw_date.split("-")
    if len(parts) != 3:
        return None

    try:
        ec_year, ec_month, ec_day = [int(part) for part in parts]
        if not (2010 <= ec_year <= 2018 and 1 <= ec_month <= 13 and 1 <= ec_day <= 30):
            return None

        gc_year_start = ec_year + 7
        start_day = 12 if (ec_year - 1) % 4 == 3 else 11
        gc_start = date(gc_year_start, 9, start_day)
        return gc_start + timedelta(days=(ec_month - 1) * 30 + ec_day - 1)
    except Exception:
        return None


def parse_enrollment_date(value: object, source: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    if normalize_source(source) == "emr":
        converted = ec_to_gc(raw)
        if converted is not None:
            return converted

    raw = re.sub(r"([+-]\d{2}):?(\d{2})$", "", raw)
    raw = re.sub(r"Z$", "", raw)
    raw_date = raw.split("T")[0].split(" ")[0]

    try:
        return date.fromisoformat(raw_date)
    except ValueError:
        return None


def clean_csv(path: Path, start_date: date, end_date: date, backup: bool) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            return 0, 0

        fieldnames = list(reader.fieldnames)
        if "source" not in fieldnames or "enrollment_date" not in fieldnames:
            return 0, 0

        kept_rows = []
        removed = 0
        total = 0

        for row in reader:
            total += 1
            if normalize_source(row.get("source")) != "emr":
                kept_rows.append(row)
                continue

            enrollment_date = parse_enrollment_date(
                row.get("enrollment_date"), row.get("source")
            )
            if enrollment_date is None or enrollment_date < start_date or enrollment_date > end_date:
                removed += 1
                continue

            kept_rows.append(row)

    if removed == 0:
        return total, removed

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)
        temp_path = Path(outfile.name)

    temp_path.replace(path)
    return total, removed


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_maternal_dir = script_dir / "imnid" / "maternal"

    parser = argparse.ArgumentParser(
        description="Clean EMR-only maternal rows outside the enrollment date window."
    )
    parser.add_argument(
        "--maternal-dir",
        type=Path,
        default=default_maternal_dir,
        help="Maternal CSV folder.",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE.isoformat(),
        help="Earliest allowed enrollment_date for EMR rows.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Latest allowed enrollment_date for EMR rows.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files before modifying CSVs.",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")

    if not args.maternal_dir.exists():
        print(f"Maternal folder not found: {args.maternal_dir}")
        return 1

    total_files = 0
    total_rows = 0
    total_removed = 0

    for csv_path in sorted(args.maternal_dir.glob("*.csv")):
        rows, removed = clean_csv(
            csv_path,
            start_date=start_date,
            end_date=end_date,
            backup=not args.no_backup,
        )
        total_files += 1
        total_rows += rows
        total_removed += removed
        print(f"{csv_path.name}: rows={rows}, removed_emr_rows={removed}")

    print(
        f"Done. Files={total_files}, rows_seen={total_rows}, "
        f"removed_emr_rows={total_removed}, window={start_date}..{end_date}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

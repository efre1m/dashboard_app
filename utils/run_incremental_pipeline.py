#!/usr/bin/env python3
"""
Incremental DHIS2 pipeline runner.

Fetches only newly-enrolled TEIs since the most recent enrollment_date timestamp
found in existing national files (or a stored state file), then appends new rows to:
  - utils/imnid/maternal/national_maternal.csv + regional_*.csv
  - utils/imnid/newborn/national_newborn.csv + regional_*.csv

Notes:
- If no baseline exists for a program, it will fetch ALL TEIs for that program.
- Deduplication is done by tei_id against existing national files.
- Time-aware filtering is applied client-side, even if the API uses date-only filters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to path to import from config / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhis2_fetcher import (  # noqa: E402
    AutomatedDHIS2Pipeline,
    CSVIntegration,
    DEFAULT_OUTPUT_DIR,
    DHIS2DataFetcher,
    MATERNAL_PROGRAM_UID,
    NEWBORN_PROGRAM_UID,
)


STATE_FILENAME = ".incremental_state.json"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        # Accept full ISO timestamps or YYYY-MM-DD
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return _normalize_dt(parsed)
    except ValueError:
        try:
            parsed = datetime.strptime(raw[:10], "%Y-%m-%d")
            return _normalize_dt(parsed)
        except ValueError:
            return None


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _format_date_for_api(
    d: Optional[date], lookback_days: int = 0
) -> Optional[str]:
    if d is None:
        return None
    if lookback_days:
        d = d - timedelta(days=lookback_days)
    return d.isoformat()


def _read_state(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _max_enrollment_dt_from_file(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["enrollment_date"], dtype=str, keep_default_na=False)
    except Exception:
        return None
    if df.empty or "enrollment_date" not in df.columns:
        return None
    ts = pd.to_datetime(df["enrollment_date"], errors="coerce", utc=True)
    if ts.dropna().empty:
        return None
    return _normalize_dt(ts.max().to_pydatetime())


def _max_enrollment_dt_from_df(df: pd.DataFrame) -> Optional[datetime]:
    if df.empty or "enrollment_date" not in df.columns:
        return None
    ts = pd.to_datetime(df["enrollment_date"], errors="coerce", utc=True)
    if ts.dropna().empty:
        return None
    return _normalize_dt(ts.max().to_pydatetime())


def _read_existing_tei_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["tei_id"], dtype=str, keep_default_na=False)
        return set(df["tei_id"].astype(str).str.strip())
    except Exception:
        return set()


def _clean_region_name(region_name: str) -> str:
    clean_region = re.sub(r"[^\w\s-]", "", region_name)
    clean_region = re.sub(r"[-\s]+", "_", clean_region)
    return clean_region


def _region_filename(region_name: str, program_type: str) -> str:
    clean_region = _clean_region_name(region_name)
    return f"regional_{clean_region}_{program_type}.csv"


def _append_rows(
    *,
    existing_path: Path,
    new_df: pd.DataFrame,
    drop_region_cols: bool,
    dry_run: bool,
) -> int:
    if new_df.empty:
        return 0

    df_to_add = new_df.copy()
    if drop_region_cols:
        df_to_add = df_to_add.drop(columns=["region_uid", "region_name"], errors="ignore")

    if dry_run:
        return len(df_to_add)

    if not existing_path.exists():
        df_to_add.to_csv(existing_path, index=False, encoding="utf-8")
        return len(df_to_add)

    existing_df = pd.read_csv(existing_path, dtype=str, keep_default_na=False)

    # Union columns, keep existing order first
    existing_cols = list(existing_df.columns)
    new_cols = [c for c in df_to_add.columns if c not in existing_cols]
    all_cols = existing_cols + new_cols

    existing_df = existing_df.reindex(columns=all_cols, fill_value="N/A")
    df_to_add = df_to_add.reindex(columns=all_cols, fill_value="N/A")

    combined = pd.concat([existing_df, df_to_add], ignore_index=True)
    combined.to_csv(existing_path, index=False, encoding="utf-8")
    return len(df_to_add)


@dataclass
class ProgramResult:
    added_rows: int
    max_enrollment_date: Optional[datetime]


def _tei_enrollment_datetime(tei: Dict) -> Optional[datetime]:
    enrollments = tei.get("enrollments") or []
    if not enrollments:
        return None
    enrollment_date = enrollments[0].get("enrollmentDate")
    if not enrollment_date:
        return None
    try:
        parsed = datetime.fromisoformat(str(enrollment_date).replace("Z", "+00:00"))
        return _normalize_dt(parsed)
    except ValueError:
        try:
            parsed = datetime.strptime(str(enrollment_date)[:10], "%Y-%m-%d")
            return _normalize_dt(parsed)
        except ValueError:
            return None


def _process_program(
    *,
    program_uid: str,
    program_name: str,
    program_type: str,
    since_date: Optional[datetime],
    output_dir: Path,
    fetcher: DHIS2DataFetcher,
    orgunit_names: Dict[str, str],
    regions: Dict[str, str],
    csv_data: Optional[pd.DataFrame],
    facility_map: Optional[Dict[str, str]],
    existing_teis: Set[str],
    lookback_days: int,
    dry_run: bool,
) -> ProgramResult:
    program_start_date = _format_date_for_api(
        since_date.date() if since_date else None, lookback_days=lookback_days
    )
    if program_start_date:
        print(f"[{program_name}] programStartDate = {program_start_date}")
    else:
        print(f"[{program_name}] No programStartDate filter (full fetch)")

    total_added = 0
    new_chunks = []

    for idx, (region_uid, region_name) in enumerate(regions.items(), start=1):
        print(f"[{program_name}] Region {idx}/{len(regions)}: {region_name}")

        tei_data = fetcher.fetch_program_data(
            program_uid,
            region_uid,
            "DESCENDANTS",
            1000,
            program_start_date=program_start_date,
        )

        teis = tei_data.get("trackedEntityInstances", [])
        if since_date:
            filtered = []
            for tei in teis:
                tei_dt = _tei_enrollment_datetime(tei)
                # Use >= so we don't miss TEIs that share the same timestamp
                # as the latest recorded enrollment_date.
                if tei_dt is None or tei_dt >= since_date:
                    filtered.append(tei)
            teis = filtered
        if existing_teis:
            teis = [
                tei
                for tei in teis
                if tei.get("trackedEntityInstance") not in existing_teis
            ]
            tei_data = {"trackedEntityInstances": teis}

        if not teis:
            print(f"[{program_name}]   No new TEIs in {region_name}")
            continue

        events_df = CSVIntegration.create_events_dataframe(
            tei_data, program_uid, orgunit_names
        )

        if program_uid == MATERNAL_PROGRAM_UID and csv_data is not None:
            region_csv = CSVIntegration.filter_csv_data_by_user_access(
                csv_data,
                "regional",
                region_name,
                facility_map,
            )
            if not region_csv.empty:
                events_df = CSVIntegration.integrate_maternal_csv_data_for_region(
                    events_df, region_csv, region_name
                )

        patient_df = CSVIntegration.transform_events_to_patient_level(
            events_df, program_uid
        )
        if patient_df.empty:
            continue

        patient_df = CSVIntegration.post_process_dataframe(patient_df)
        patient_df["region_uid"] = region_uid
        patient_df["region_name"] = region_name

        # Deduplicate against existing TEIs again after post-processing
        new_df = patient_df[
            ~patient_df["tei_id"].astype(str).str.strip().isin(existing_teis)
        ]

        if new_df.empty:
            print(f"[{program_name}]   No new rows after dedupe in {region_name}")
            continue

        region_file = output_dir / _region_filename(region_name, program_type)
        added = _append_rows(
            existing_path=region_file,
            new_df=new_df,
            drop_region_cols=True,
            dry_run=dry_run,
        )
        total_added += added

        existing_teis.update(new_df["tei_id"].astype(str).str.strip())
        new_chunks.append(new_df)

        print(f"[{program_name}]   Added {added} rows to {region_file.name}")

    if not new_chunks:
        return ProgramResult(added_rows=0, max_enrollment_date=None)

    combined_new = pd.concat(new_chunks, ignore_index=True)
    national_file = output_dir / f"national_{program_type}.csv"
    _append_rows(
        existing_path=national_file,
        new_df=combined_new,
        drop_region_cols=False,
        dry_run=dry_run,
    )

    max_enrollment = _max_enrollment_dt_from_df(combined_new)
    return ProgramResult(added_rows=total_added, max_enrollment_date=max_enrollment)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incremental DHIS2 pipeline (append new enrollments only)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (default: utils/imnid).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional maternal CSV path for integration.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Override start date for BOTH programs (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental state JSON (default: <output-dir>/.incremental_state.json).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=2,
        help="Subtract N days from start date to avoid missing late enrollments (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only report what would be appended.",
    )
    args = parser.parse_args()

    load_dotenv()

    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    maternal_dir = output_dir / "maternal"
    newborn_dir = output_dir / "newborn"
    maternal_dir.mkdir(parents=True, exist_ok=True)
    newborn_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_file) if args.state_file else output_dir / STATE_FILENAME
    state = _read_state(state_path)

    override_since = _parse_iso_datetime(args.since)
    maternal_since = override_since
    newborn_since = override_since

    if override_since is None:
        maternal_since = _parse_iso_datetime(state.get("maternal_last_enrollment_date"))
        newborn_since = _parse_iso_datetime(state.get("newborn_last_enrollment_date"))

    if maternal_since is None:
        maternal_since = _max_enrollment_dt_from_file(
            maternal_dir / "national_maternal.csv"
        )
    if newborn_since is None:
        newborn_since = _max_enrollment_dt_from_file(
            newborn_dir / "national_newborn.csv"
        )

    pipeline = AutomatedDHIS2Pipeline(
        base_url=None,
        username=None,
        password=None,
        csv_path=args.csv_path,
        output_base_dir=str(output_dir),
    )
    if not pipeline.base_url or not pipeline.username or not pipeline.password:
        print("Missing DHIS2 credentials. Check your .env or config.py settings.")
        return 1

    fetcher = pipeline.fetcher

    # Incremental mode (always)
    if args.csv_path:
        pipeline.load_csv_data()
    orgunit_names = fetcher.fetch_orgunit_names()
    regions = fetcher.fetch_all_regions()
    if not regions:
        print("No regions found. Check DHIS2 credentials/connection.")
        return 1

    existing_maternal_teis = _read_existing_tei_ids(
        maternal_dir / "national_maternal.csv"
    )
    existing_newborn_teis = _read_existing_tei_ids(
        newborn_dir / "national_newborn.csv"
    )

    print("=" * 72)
    print("INCREMENTAL PIPELINE")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Maternal since: {maternal_since}")
    print(f"Newborn since: {newborn_since}")
    print("=" * 72)

    maternal_result = _process_program(
        program_uid=MATERNAL_PROGRAM_UID,
        program_name="MATERNAL",
        program_type="maternal",
        since_date=maternal_since,
        output_dir=maternal_dir,
        fetcher=fetcher,
        orgunit_names=orgunit_names,
        regions=regions,
        csv_data=pipeline.csv_data,
        facility_map=pipeline.facility_to_region_map,
        existing_teis=existing_maternal_teis,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
    )

    newborn_result = _process_program(
        program_uid=NEWBORN_PROGRAM_UID,
        program_name="NEWBORN",
        program_type="newborn",
        since_date=newborn_since,
        output_dir=newborn_dir,
        fetcher=fetcher,
        orgunit_names=orgunit_names,
        regions=regions,
        csv_data=None,  # no CSV integration for newborn
        facility_map=None,
        existing_teis=existing_newborn_teis,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
    )

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Maternal added rows: {maternal_result.added_rows}")
    print(f"Newborn added rows: {newborn_result.added_rows}")

    if not args.dry_run:
        if maternal_result.max_enrollment_date:
            state["maternal_last_enrollment_date"] = (
                maternal_result.max_enrollment_date.isoformat()
            )
        if newborn_result.max_enrollment_date:
            state["newborn_last_enrollment_date"] = (
                newborn_result.max_enrollment_date.isoformat()
            )
        state["last_run_utc"] = datetime.utcnow().isoformat() + "Z"
        _write_state(state_path, state)
        print(f"State file updated: {state_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

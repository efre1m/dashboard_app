#!/usr/bin/env python3
"""
Incremental DHIS2 pipeline runner.

Fetches TEIs in a recent window and upserts them (inserts new TEIs and refreshes
existing TEIs), preferring the last successful run timestamp and falling back to
enrollment-based windows only when no prior run state exists, then writes to:
  - utils/imnid/maternal/national_maternal.csv + regional_*.csv
  - utils/imnid/newborn/national_newborn.csv + regional_*.csv

Notes:
- If no baseline exists for a program, it will fetch ALL TEIs for that program.
- Deduplication/upsert is done by tei_id against existing files.
- Regular incremental runs use lastUpdated windows so delayed offline syncs still refresh stage values.
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
from typing import Dict, Optional, Set

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


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILENAME = ".incremental_state.json"
DEFAULT_MATERNAL_CSV = "maternal_data_long_format.csv"


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


def _format_datetime_for_api(
    dt: Optional[datetime], lookback_days: int = 0
) -> Optional[str]:
    if dt is None:
        return None
    if lookback_days:
        dt = dt - timedelta(days=lookback_days)
    return dt.replace(microsecond=0).isoformat()


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


def _reprocess_saved_file(path: Path, dry_run: bool) -> None:
    if dry_run or not path.exists():
        return
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if df.empty:
            return
        processed = CSVIntegration.post_process_dataframe(df)
        processed.to_csv(path, index=False, encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to reprocess {path.name}: {exc}")


def _log_fetched_event_values(
    *,
    program_name: str,
    region_name: str,
    events_df: pd.DataFrame,
    limit: int = 5,
) -> None:
    if events_df.empty:
        print(f"[{program_name}]   No event rows created for {region_name}")
        return

    actual_rows = events_df.copy()
    if "has_actual_event" in actual_rows.columns:
        actual_rows = actual_rows[actual_rows["has_actual_event"] == True]

    if "value" in actual_rows.columns:
        actual_rows = actual_rows[
            actual_rows["value"].astype(str).str.strip().ne("")
        ]

    value_count = len(actual_rows)
    print(f"[{program_name}]   Event values fetched for {region_name}: {value_count}")

    if value_count == 0:
        return

    sample_cols = ["tei_id", "programStageName", "dataElementName", "value"]
    sample = (
        actual_rows[sample_cols]
        .drop_duplicates()
        .head(limit)
        .fillna("")
    )
    for _, row in sample.iterrows():
        print(
            f"[{program_name}]     {row['tei_id']} | {row['programStageName']} | "
            f"{row['dataElementName']} = {row['value']}"
        )


def _upsert_rows(
    *,
    existing_path: Path,
    new_df: pd.DataFrame,
    key_col: str,
    drop_region_cols: bool,
    dry_run: bool,
) -> tuple[int, int]:
    if new_df.empty:
        return 0, 0

    df_to_add = new_df.copy()
    if drop_region_cols:
        df_to_add = df_to_add.drop(
            columns=["region_uid", "region_name"], errors="ignore"
        )

    if key_col not in df_to_add.columns:
        raise KeyError(f"Missing key column {key_col!r} in new_df")

    df_to_add[key_col] = df_to_add[key_col].astype(str).str.strip()
    new_keys = set(df_to_add[key_col].tolist())

    if dry_run:
        if not existing_path.exists():
            return len(new_keys), 0
        try:
            existing_df = pd.read_csv(existing_path, usecols=[key_col], dtype=str, keep_default_na=False)
            existing_df[key_col] = existing_df[key_col].astype(str).str.strip()
            existing_keys = set(existing_df[key_col].tolist())
        except Exception:
            existing_keys = set()
        updated = len(new_keys & existing_keys)
        inserted = len(new_keys - existing_keys)
        return inserted, updated

    if not existing_path.exists():
        df_to_add.to_csv(existing_path, index=False, encoding="utf-8")
        return len(new_keys), 0

    existing_df = pd.read_csv(existing_path, dtype=str, keep_default_na=False)
    if key_col not in existing_df.columns:
        raise KeyError(f"Missing key column {key_col!r} in {existing_path}")

    existing_df[key_col] = existing_df[key_col].astype(str).str.strip()
    existing_keys = set(existing_df[key_col].tolist())
    updated = len(new_keys & existing_keys)
    inserted = len(new_keys - existing_keys)

    # Ensure unique keys before merging (prefer latest occurrence).
    existing_df = existing_df.drop_duplicates(subset=[key_col], keep="last")
    df_to_add = df_to_add.drop_duplicates(subset=[key_col], keep="last")

    # Union columns, keep existing order first
    existing_cols = list(existing_df.columns)
    new_cols = [c for c in df_to_add.columns if c not in existing_cols]
    all_cols = existing_cols + new_cols

    existing_df = existing_df.reindex(columns=all_cols, fill_value="N/A")
    df_to_add = df_to_add.reindex(columns=all_cols, fill_value="N/A")

    def _is_missing(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        return s.isin({"", "n/a", "nan", "none"})

    # Merge without destroying existing values:
    # - If incoming value is non-missing, overwrite existing.
    # - If incoming is missing (N/A/empty), keep existing.
    existing_idx = existing_df.set_index(key_col)
    incoming_idx = df_to_add.set_index(key_col)

    overlap = existing_idx.index.intersection(incoming_idx.index)
    new_only = incoming_idx.index.difference(existing_idx.index)

    if not overlap.empty:
        for col in all_cols:
            if col == key_col:
                continue
            inc = incoming_idx.loc[overlap, col]
            non_missing = ~_is_missing(inc)
            existing_idx.loc[overlap, col] = existing_idx.loc[overlap, col].where(
                ~non_missing, inc
            )

    combined = (
        pd.concat([existing_idx, incoming_idx.loc[new_only]], axis=0)
        .reset_index()
        .reindex(columns=all_cols, fill_value="N/A")
    )
    combined.to_csv(existing_path, index=False, encoding="utf-8")
    return inserted, updated


@dataclass
class ProgramResult:
    fetched_rows: int
    inserted_rows: int
    updated_rows: int
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
    last_updated_since: Optional[datetime],
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
    program_start_date = None
    last_updated_start_date = None

    if last_updated_since is not None:
        last_updated_start_date = _format_datetime_for_api(
            last_updated_since, lookback_days=lookback_days
        )
        print(f"[{program_name}] lastUpdatedStartDate = {last_updated_start_date}")
    else:
        program_start_date = _format_date_for_api(
            since_date.date() if since_date else None, lookback_days=lookback_days
        )
        if program_start_date:
            print(f"[{program_name}] programStartDate = {program_start_date}")
        else:
            print(f"[{program_name}] No incremental filter (full fetch)")

    if last_updated_start_date:
        print(
            f"[{program_name}] Using lastUpdated window so recently synced stage values are included"
        )
    elif program_start_date:
        print(
            f"[{program_name}] Using enrollment window fallback because no previous run timestamp exists"
        )
    else:
        print(f"[{program_name}] No previous state found; fetching all available TEIs")

    total_inserted = 0
    total_updated = 0
    total_fetched = 0
    changed_chunks = []

    for idx, (region_uid, region_name) in enumerate(regions.items(), start=1):
        print(f"[{program_name}] Region {idx}/{len(regions)}: {region_name}")

        tei_data = fetcher.fetch_program_data(
            program_uid,
            region_uid,
            "DESCENDANTS",
            1000,
            program_start_date=program_start_date,
            last_updated_start_date=last_updated_start_date,
        )

        teis = tei_data.get("trackedEntityInstances", [])
        tei_data = {"trackedEntityInstances": teis}

        if not teis:
            print(f"[{program_name}]   No TEIs in window for {region_name}")
            continue

        total_fetched += len(teis)
        nested_events = 0
        for tei in teis:
            for enrollment in tei.get("enrollments", []) or []:
                nested_events += len(enrollment.get("events", []) or [])
        print(
            f"[{program_name}]   TEIs fetched from DHIS2 for {region_name}: {len(teis)} "
            f"| nested events: {nested_events}"
        )

        events_df = CSVIntegration.create_events_dataframe(
            tei_data, program_uid, orgunit_names
        )
        _log_fetched_event_values(
            program_name=program_name,
            region_name=region_name,
            events_df=events_df,
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

        region_file = output_dir / _region_filename(region_name, program_type)
        inserted, updated = _upsert_rows(
            existing_path=region_file,
            new_df=patient_df,
            key_col="tei_id",
            drop_region_cols=True,
            dry_run=dry_run,
        )
        total_inserted += inserted
        total_updated += updated
        _reprocess_saved_file(region_file, dry_run)

        existing_teis.update(patient_df["tei_id"].astype(str).str.strip())
        changed_chunks.append(patient_df)

        print(
            f"[{program_name}]   Region summary for {region_name}: fetched={len(teis)}, "
            f"new={inserted}, updated={updated}"
        )
        print(
            f"[{program_name}]   Upserted {inserted + updated} rows to {region_file.name} "
            f"(new={inserted}, updated={updated})"
        )

    if not changed_chunks:
        return ProgramResult(
            fetched_rows=total_fetched,
            inserted_rows=0,
            updated_rows=0,
            max_enrollment_date=None,
        )

    combined_changed = pd.concat(changed_chunks, ignore_index=True)
    national_file = output_dir / f"national_{program_type}.csv"
    _upsert_rows(
        existing_path=national_file,
        new_df=combined_changed,
        key_col="tei_id",
        drop_region_cols=False,
        dry_run=dry_run,
    )
    _reprocess_saved_file(national_file, dry_run)

    max_enrollment = _max_enrollment_dt_from_df(combined_changed)
    print(
        f"[{program_name}] TOTAL fetched from DHIS2: {total_fetched} | "
        f"new={total_inserted} | updated={total_updated}"
    )
    return ProgramResult(
        fetched_rows=total_fetched,
        inserted_rows=total_inserted,
        updated_rows=total_updated,
        max_enrollment_date=max_enrollment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incremental DHIS2 pipeline (upsert recent TEIs)."
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
        help="Override incremental start datetime for BOTH programs (ISO format, for example 2026-04-13T10:15:00).",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental state JSON (default: <output-dir>/.incremental_state.json).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Subtract N days from the incremental fetch boundary to catch delayed offline syncs (default: 14).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only report what would be upserted.",
    )
    args = parser.parse_args()

    # Always prefer the repo .env for this command-line runner. This avoids
    # stale process or Windows environment variables shadowing recently edited
    # DHIS2 credentials.
    load_dotenv(REPO_ROOT / ".env", override=True)

    csv_path = args.csv_path
    if csv_path is None and Path(DEFAULT_MATERNAL_CSV).exists():
        csv_path = DEFAULT_MATERNAL_CSV

    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    maternal_dir = output_dir / "maternal"
    newborn_dir = output_dir / "newborn"
    maternal_dir.mkdir(parents=True, exist_ok=True)
    newborn_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_file) if args.state_file else output_dir / STATE_FILENAME
    state = _read_state(state_path)

    override_since = _parse_iso_datetime(args.since)
    last_run_since = override_since or _parse_iso_datetime(state.get("last_run_utc"))
    maternal_since = None
    newborn_since = None

    if last_run_since is None:
        maternal_since = _parse_iso_datetime(state.get("maternal_last_enrollment_date"))
        newborn_since = _parse_iso_datetime(state.get("newborn_last_enrollment_date"))

    if maternal_since is None and last_run_since is None:
        maternal_since = _max_enrollment_dt_from_file(
            maternal_dir / "national_maternal.csv"
        )
    if newborn_since is None and last_run_since is None:
        newborn_since = _max_enrollment_dt_from_file(
            newborn_dir / "national_newborn.csv"
        )

    pipeline = AutomatedDHIS2Pipeline(
        base_url=os.getenv("DHIS2_BASE_URL"),
        username=os.getenv("DHIS2_USERNAME"),
        password=os.getenv("DHIS2_PASSWORD"),
        csv_path=csv_path,
        output_base_dir=str(output_dir),
    )
    if not pipeline.base_url or not pipeline.username or not pipeline.password:
        print("Missing DHIS2 credentials. Check your .env or config.py settings.")
        return 1

    fetcher = pipeline.fetcher

    # Incremental mode (always)
    if pipeline.csv_path:
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
    print(f"Maternal CSV integration: {pipeline.csv_path or 'disabled'}")
    print(f"Last run timestamp: {last_run_since}")
    print(f"Maternal enrollment fallback: {maternal_since}")
    print(f"Newborn enrollment fallback: {newborn_since}")
    print("=" * 72)

    maternal_result = _process_program(
        program_uid=MATERNAL_PROGRAM_UID,
        program_name="MATERNAL",
        program_type="maternal",
        since_date=maternal_since,
        last_updated_since=last_run_since,
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
        last_updated_since=last_run_since,
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
    print(
        f"Maternal fetched from DHIS2: {maternal_result.fetched_rows} "
        f"(new={maternal_result.inserted_rows}, updated={maternal_result.updated_rows})"
    )
    print(
        f"Maternal upserted: {maternal_result.inserted_rows + maternal_result.updated_rows} "
        f"(new={maternal_result.inserted_rows}, updated={maternal_result.updated_rows})"
    )
    print(
        f"Newborn fetched from DHIS2: {newborn_result.fetched_rows} "
        f"(new={newborn_result.inserted_rows}, updated={newborn_result.updated_rows})"
    )
    print(
        f"Newborn upserted: {newborn_result.inserted_rows + newborn_result.updated_rows} "
        f"(new={newborn_result.inserted_rows}, updated={newborn_result.updated_rows})"
    )
    print(
        f"Grand total fetched from DHIS2: "
        f"{maternal_result.fetched_rows + newborn_result.fetched_rows}"
    )

    if not args.dry_run:
        if maternal_result.max_enrollment_date:
            prev = _parse_iso_datetime(state.get("maternal_last_enrollment_date"))
            if prev is None or maternal_result.max_enrollment_date > prev:
                state["maternal_last_enrollment_date"] = (
                    maternal_result.max_enrollment_date.isoformat()
                )
        if newborn_result.max_enrollment_date:
            prev = _parse_iso_datetime(state.get("newborn_last_enrollment_date"))
            if prev is None or newborn_result.max_enrollment_date > prev:
                state["newborn_last_enrollment_date"] = (
                    newborn_result.max_enrollment_date.isoformat()
                )
        state["last_run_utc"] = datetime.utcnow().isoformat() + "Z"
        _write_state(state_path, state)
        print(f"State file updated: {state_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

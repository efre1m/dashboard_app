import os
import sys
import traceback
import argparse
from datetime import date, datetime, timedelta
from typing import List

import pandas as pd
from dotenv import load_dotenv

from add_source_column import assign_source_column
import dhis2_fetcher as core
from dhis2_fetcher import (
    CSVIntegration,
    DHIS2DataFetcher,
    DEFAULT_OUTPUT_DIR,
    logger,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NID_PROGRAM_UID = "pLk3Ht2XMKl"

ADDITIONAL_O2_DATAELEMENT_UID = "j4W59YyYG04"
ADDITIONAL_O2_DATAELEMENT_NAME = "Lowest recorded oxygen saturation (%)"
PHOTOTHERAPY_DATAELEMENT_UID = "EDwkmS1fKbG"
PHOTOTHERAPY_DATAELEMENT_NAME = "Was Phototherapy Administered?"
TRANSFUSION_DATAELEMENT_UID = "zRsdb6F5JpB"
TRANSFUSION_DATAELEMENT_NAME = "Was a transfusion given?"
BILIRUBIN_DATAELEMENT_UID = "U7ey9R5QZk1"
BILIRUBIN_DATAELEMENT_NAME = "Was bilirubin tested?"
OBS_STAGE_UID = "VsVlpG1V2ub"  # Observations And Nursing Care 2

NID_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "nid")

NATIONAL_OUTPUT_FILE = "national_newborn_nid.csv"

EXPECTED_NEWBORN_STAGE_MAPPING = {
    "l39SlVGlQGs": {
        "data_elements": [
            "UOmhJkyAK6h",
            "yxWUMt3sCil",
            "T30GbTiVgFR",
            "OpHw2X58x5i",
            "gJH6PkYI6IV",
            "aK5txmRYpVX",
            "p2GxXHvzlnC",
            "K5hp1PVB5l4",
        ],
        "program_stage_name": "Admission Information",
    },
    "UK7jsDbVpj6": {
        "data_elements": ["CzIgD0rsk52"],
        "program_stage_name": "Maternal Birth And Infant Details",
    },
    "j0HI2eJjvbj": {
        "data_elements": ["gZi9y12E9i7", "kvJUcoHhx7f", "tX82J8ZIcNr"],
        "program_stage_name": "Observations And Nursing Care 1",
    },
    "ed8ErpgTCwx": {
        "data_elements": ["QK7Fi6OwtDC", "wlHEf9FdmJM", "sxtsEDilKZd", "yVNrsnUo7AR", "Ul6aAlFy5Kk", "CIKlewzNAxj", "JxGlnPychB0", PHOTOTHERAPY_DATAELEMENT_UID, TRANSFUSION_DATAELEMENT_UID],
        "program_stage_name": "Interventions",
    },
    "TOicTEwzSGj": {
        "data_elements": [
            "vmOAGuFcaz4", "yBCwmQP0A6a", "wn0tHaHcceW",
            "X2m8NB1P83P", "EClz4eZuxH2", "VCatFN4N9wD",
            "o1sbIt5YJ8b", "t2iHoRMo5hn",
        ],
        "program_stage_name": "Discharge And Final Diagnosis",
    },
    "VsVlpG1V2ub": {
        "data_elements": ["nIKIu6f5vbW", ADDITIONAL_O2_DATAELEMENT_UID, BILIRUBIN_DATAELEMENT_UID],
        "program_stage_name": "Observations And Nursing Care 2",
    },
    "aCrttmnx7FI": {
        "data_elements": ["A94ibeuO9GL"],
        "program_stage_name": "Microbiology And Labs",
    },
}

EXPECTED_NEWBORN_DATAELEMENT_NAMES = {
    "QK7Fi6OwtDC": "KMC Administered",
    "yxWUMt3sCil": "Weight on admission",
    "gZi9y12E9i7": "Temperature on admission (°C)",
    "UOmhJkyAK6h": "Date of Admission",
    "wlHEf9FdmJM": "CPAP Administered",
    "T30GbTiVgFR": "First Reason for Admission",
    "OpHw2X58x5i": "Second Reason for Admission",
    "gJH6PkYI6IV": "Third Reason for Admission",
    "aK5txmRYpVX": "birth location",
    "vmOAGuFcaz4": "Newborn Status at Discharge",
    "yBCwmQP0A6a": "Discharge Weight (grams)",
    "nIKIu6f5vbW": "lowest recorded temperature (Celsius)",
    "sxtsEDilKZd": "Were antibiotics administered?",
    "X2m8NB1P83P": "Primary Category",
    "EClz4eZuxH2": "Sub-Categories of Congenital Malformations",
    "VCatFN4N9wD": "Sub-Categories of Prematurity",
    "wn0tHaHcceW": "Sub-Categories of Infection",
    "o1sbIt5YJ8b": "Sub-Categories of Intrapartum-Related",
    "t2iHoRMo5hn": "Sub-Categories of Jaundice (Pathological)",
    "A94ibeuO9GL": "Blood culture for suspected sepsis",
    "CzIgD0rsk52": "Birth weight (grams)",
    "p2GxXHvzlnC": "Time of Birth",
    "K5hp1PVB5l4": "Time of Admission",
    "yVNrsnUo7AR": "CPAP (1) Start Date",
    "Ul6aAlFy5Kk": "CPAP (1) Start Time",
    "CIKlewzNAxj": "Type of CPAP machine used:",
    "kvJUcoHhx7f": "Was oxygen saturation (%) recorded on admission?",
    "tX82J8ZIcNr": "Was blood sugar recorded on admission?",
    "JxGlnPychB0": "If yes - KMC Start Date:",
    ADDITIONAL_O2_DATAELEMENT_UID: ADDITIONAL_O2_DATAELEMENT_NAME,
    PHOTOTHERAPY_DATAELEMENT_UID: PHOTOTHERAPY_DATAELEMENT_NAME,
    TRANSFUSION_DATAELEMENT_UID: TRANSFUSION_DATAELEMENT_NAME,
    BILIRUBIN_DATAELEMENT_UID: BILIRUBIN_DATAELEMENT_NAME,
}


def prompt_enrollment_date_filter() -> dict:
    """Prompt user for enrollment date filter and return start/end date strings."""
    print("\nSelect enrollment date filter mode:")
    print("  1. Between two dates")
    print("  2. Before a date (exclusive)")
    print("  3. After a date (exclusive)")
    print("  4. No filter")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        start = input("Enter start date (YYYY-MM-DD): ").strip()
        end = input("Enter end date (YYYY-MM-DD): ").strip()
        print(f"  => Enrollment Date >= {start}, Enrollment Date <= {end}")
        return {"program_start_date": start, "program_end_date": end, "mode": "between"}
    elif choice == "2":
        before = input("Enter before date (YYYY-MM-DD) – enrollments strictly before this date: ").strip()
        # Shift one day back so DHIS2 inclusive filter effectively excludes the given date
        dt = datetime.strptime(before, "%Y-%m-%d") - timedelta(days=1)
        end = dt.strftime("%Y-%m-%d")
        print(f"  => Enrollment Date < {before} (API: Enrollment Date <= {end})")
        return {"program_start_date": None, "program_end_date": end, "mode": "before"}
    elif choice == "3":
        after = input("Enter after date (YYYY-MM-DD) – enrollments strictly after this date: ").strip()
        # Shift one day forward so DHIS2 inclusive filter effectively excludes the given date
        dt = datetime.strptime(after, "%Y-%m-%d") + timedelta(days=1)
        start = dt.strftime("%Y-%m-%d")
        print(f"  => Enrollment Date > {after} (API: Enrollment Date >= {start})")
        return {"program_start_date": start, "program_end_date": None, "mode": "after"}
    else:
        print("  => No enrollment date filter")
        return {"program_start_date": None, "program_end_date": None, "mode": "none"}


def regional_output_filename(region_name: str) -> str:
    clean_region = "".join(ch if ch.isalnum() or ch in " _-" else "" for ch in str(region_name))
    clean_region = "_".join(clean_region.replace("-", " ").split())
    return f"regional_{clean_region}_newborn_nid.csv"


def ensure_additional_dataelement_mapping() -> None:
    """Force exact newborn mappings + additional oxygen saturation at runtime."""
    core.NEWBORN_PROGRAM_STAGE_MAPPING = EXPECTED_NEWBORN_STAGE_MAPPING
    core.DATA_ELEMENT_NAMES.update(EXPECTED_NEWBORN_DATAELEMENT_NAMES)
    for de_uid in EXPECTED_NEWBORN_DATAELEMENT_NAMES:
        core.NEWBORN_HEALTH_ELEMENTS.add(de_uid)
    core.REQUIRED_DATA_ELEMENTS = core.MATERNAL_HEALTH_ELEMENTS | core.NEWBORN_HEALTH_ELEMENTS
    logger.info("Applied expected newborn stage mapping and data element names (including oxygen saturation)")


def preprocess_nid_enrollment_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardize enrollment dates that appear to be entered in Ethiopian Calendar.

    For rows where enrollment_date < 2025-01-01:
    1. Look for a valid date >= 2025-01-01 in any event_date_* column
    2. If found, replace enrollment_date with that event date value
    3. If no valid event date is found, try Ethiopian -> Gregorian conversion
    4. If the converted date is in the future (> today), remove the row and log tei_id

    Returns (cleaned_df, list_of_removed_tei_ids).
    """
    if df.empty or "enrollment_date" not in df.columns:
        return df, []

    df = df.copy()
    removed_teis: List[str] = []
    event_date_cols = [c for c in df.columns if c.startswith("event_date_")]
    cutoff = date(2025, 1, 1)

    def _parse(val: str):
        if not val or val in ("N/A", "nan", ""):
            return None
        try:
            v = val.split("T")[0].split(" ")[0]
            p = v.split("-")
            if len(p) != 3:
                return None
            return date(int(p[0]), int(p[1]), int(p[2]))
        except Exception:
            return None

    for idx, row in df.iterrows():
        raw_enc = str(row.get("enrollment_date", "")).strip()
        if not raw_enc or raw_enc in ("N/A", "nan", ""):
            continue

        enc_date = _parse(raw_enc)
        if enc_date is None:
            continue

        # Already valid (>= Jan 2025), skip
        if enc_date >= cutoff:
            continue

        # Method 1: find a valid event_date_* >= cutoff
        found = False
        for col in event_date_cols:
            evt = _parse(str(row.get(col, "")))
            if evt is not None and evt >= cutoff:
                df.at[idx, "enrollment_date"] = row[col]
                found = True
                break
        if found:
            continue

        # Method 2: Ethiopian -> Gregorian conversion
        try:
            gy_start = enc_date.year + 7
            start_day = 12 if (enc_date.year - 1) % 4 == 3 else 11
            start_gc = date(gy_start, 9, start_day)
            days_off = (enc_date.month - 1) * 30 + (enc_date.day - 1)
            gc_date = start_gc + timedelta(days=days_off)
        except Exception:
            continue

        if gc_date > date.today():
            removed_teis.append(str(row.get("tei_id", "")))
            df.at[idx, "_remove"] = True
            continue

        df.at[idx, "enrollment_date"] = gc_date.strftime("%Y-%m-%dT00:00:00.000")

    if "_remove" in df.columns:
        n_before = len(df)
        df = df[df["_remove"] != True].drop(columns=["_remove"])
        logger.warning(
            f"Preprocess: removed {len(removed_teis)} TEI(s) with future "
            f"EC-converted enrollment date: {removed_teis}"
        )

    return df, removed_teis


def append_missing_teis(source_df: pd.DataFrame, target_path: str, merge_mode: str = "replace") -> int:
    source_df = assign_source_column(source_df)
    source_df.columns = source_df.columns.str.strip()

    if "tei_id" not in source_df.columns:
        raise ValueError("Source data missing tei_id")

    source_df["tei_id"] = source_df["tei_id"].astype(str).str.strip()
    source_df = source_df[source_df["tei_id"] != ""].copy()
    source_df = source_df.drop_duplicates(subset=["tei_id"], keep="first")

    if os.path.exists(target_path):
        target_df = pd.read_csv(target_path, dtype=str, keep_default_na=False)
        target_df.columns = target_df.columns.str.strip()

        if "tei_id" not in target_df.columns:
            raise ValueError(f"Target missing tei_id: {target_path}")

        target_df["tei_id"] = target_df["tei_id"].astype(str).str.strip()
        if merge_mode not in {"replace", "new_only"}:
            raise ValueError("merge_mode must be 'replace' or 'new_only'")

        existing_teis = set(target_df["tei_id"])
        source_teis = set(source_df["tei_id"])
        overlap_teis = existing_teis.intersection(source_teis)
        to_add = source_df[~source_df["tei_id"].isin(existing_teis)].copy()

        for col in target_df.columns:
            if col not in to_add.columns:
                to_add[col] = "N/A"
        for col in to_add.columns:
            if col not in target_df.columns:
                target_df[col] = "N/A"

        # Keep all columns (including newly fetched ones) instead of dropping extras.
        combined_columns = list(target_df.columns) + [
            c for c in to_add.columns if c not in target_df.columns
        ]
        target_df = target_df.reindex(columns=combined_columns, fill_value="N/A")
        to_add = to_add.reindex(columns=combined_columns, fill_value="N/A")
        if merge_mode == "replace":
            # Upsert behavior:
            # 1) Remove target rows for overlapping TEIs
            # 2) Add full source rows for overlapping TEIs (fresh values)
            # 3) Append brand-new TEIs
            preserved_target = target_df[~target_df["tei_id"].isin(overlap_teis)].copy()
            updated_rows = source_df[source_df["tei_id"].isin(overlap_teis)].copy()
            combined = pd.concat([preserved_target, updated_rows, to_add], ignore_index=True)
            combined = combined.drop_duplicates(subset=["tei_id"], keep="last")
            updated_count = len(updated_rows)
        else:
            # New-only behavior: keep existing TEIs unchanged, append only new TEIs.
            combined = pd.concat([target_df, to_add], ignore_index=True)
            combined = combined.drop_duplicates(subset=["tei_id"], keep="first")
            updated_count = 0

        combined = assign_source_column(combined)
        combined.to_csv(target_path, index=False, encoding="utf-8")

        mode_label = "Upserted" if merge_mode == "replace" else "Appended new"
        logger.info(
            f"{mode_label} {os.path.basename(target_path)}: "
            f"updated={updated_count}, added={len(to_add)}, total={len(combined)}"
        )
        return updated_count + len(to_add)

    source_df = assign_source_column(source_df)
    source_df.to_csv(target_path, index=False, encoding="utf-8")
    logger.info(f"Created {os.path.basename(target_path)} with {len(source_df)} TEIs")
    return len(source_df)


class AutomatedLearningFacilitiesNIDPipeline:
    def __init__(
        self,
        base_url: str = None,
        username: str = None,
        password: str = None,
        merge_mode: str = "replace",
        enrollment_date_filters: dict = None,
        org_unit_ids: list = None,
    ):
        env_base_url = os.getenv("DHIS2_BASE_URL")
        env_username = os.getenv("DHIS2_USERNAME")
        env_password = os.getenv("DHIS2_PASSWORD")

        if not all([base_url, username, password]) and all(
            [env_base_url, env_username, env_password]
        ):
            base_url = base_url or env_base_url.rstrip("/")
            username = username or env_username
            password = password or env_password
            logger.info("Using credentials from environment/.env")

        self.base_url = base_url
        self.username = username
        self.password = password
        self.merge_mode = merge_mode
        self.enrollment_date_filters = enrollment_date_filters or {}
        self.org_unit_ids = org_unit_ids or []
        self.fetcher = DHIS2DataFetcher(self.base_url, self.username, self.password)

    @staticmethod
    def _fix_o2_data(events_df: pd.DataFrame, patient_df: pd.DataFrame) -> pd.DataFrame:
        o2_mask = (
            (events_df["programStageName"] == "Observations And Nursing Care 2")
            & (events_df["dataElement_uid"] == ADDITIONAL_O2_DATAELEMENT_UID)
            & (events_df["has_actual_event"] == True)
            & (events_df["value"].notna())
            & (events_df["value"] != "")
        )
        o2_events = events_df[o2_mask].copy()
        if o2_events.empty:
            return patient_df
        o2_events = o2_events.sort_values("eventDate")
        o2_events["o2_num"] = pd.to_numeric(o2_events["value"], errors="coerce")
        o2_best = o2_events.dropna(subset=["o2_num"]).groupby("tei_id", sort=False).first().reset_index()
        if o2_best.empty:
            return patient_df
        tei_val = o2_best.set_index("tei_id")["o2_num"].to_dict()
        tei_date = o2_best.set_index("tei_id")["eventDate"].to_dict()
        tei_event = o2_best.set_index("tei_id")["event"].to_dict()
        patient_df["lowest_recorded_oxygen_saturation_pct_observations_and_nursing_care_2"] = (
            patient_df["tei_id"].map(tei_val)
        )
        patient_df["event_date_observations_and_nursing_care_2"] = (
            patient_df["tei_id"].map(tei_date)
        )
        patient_df["event_observations_and_nursing_care_2"] = (
            patient_df["tei_id"].map(tei_event)
        )
        return patient_df

    def run_pipeline(self) -> bool:
        logger.info("=" * 80)
        logger.info("STARTING LEARNING FACILITIES NID FETCH + MERGE")
        logger.info(f"Program UID: {NID_PROGRAM_UID}")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("=" * 80)

        try:
            if not all([self.base_url, self.username, self.password]):
                logger.error("Missing DHIS2 credentials (DHIS2_BASE_URL, DHIS2_USERNAME, DHIS2_PASSWORD)")
                return False

            ensure_additional_dataelement_mapping()
            os.makedirs(NID_DIR, exist_ok=True)

            logger.info("Fetching orgUnit names...")
            orgunit_names = self.fetcher.fetch_orgunit_names()
            logger.info(f"Fetched {len(orgunit_names)} orgUnit names")

            prog_start = self.enrollment_date_filters.get("program_start_date")
            prog_end = self.enrollment_date_filters.get("program_end_date")
            mode = self.enrollment_date_filters.get("mode", "none")
            logger.info(f"Enrollment date filter mode: {mode}")
            if prog_start:
                logger.info(f"  program_start_date (enrollment >=): {prog_start}")
            if prog_end:
                logger.info(f"  program_end_date (enrollment <=): {prog_end}")

            all_region_patient_dfs: List[pd.DataFrame] = []
            total_fetched_teis = 0

            if self.org_unit_ids:
                logger.info(f"Fetching data for {len(self.org_unit_ids)} specific org unit(s)...")
                logger.info("Fetching facility-to-region mapping...")
                facility_to_region = self.fetcher.fetch_facility_to_region_mapping()
                regions = self.fetcher.fetch_all_regions()
                region_by_name = {v: k for k, v in regions.items()}

                for ouid in self.org_unit_ids:
                    logger.info(f"  Fetching orgUnit: {ouid}")
                    tei_data = self.fetcher.fetch_program_data(
                        NID_PROGRAM_UID, ouid, "SELECTED", 1000,
                        program_start_date=prog_start, program_end_date=prog_end,
                    )
                    tei_count = len(tei_data.get("trackedEntityInstances", []))
                    total_fetched_teis += tei_count
                    logger.info(f"  Fetched TEIs: {tei_count}")
                    if tei_count == 0:
                        continue
                    events_df = CSVIntegration.create_events_dataframe(
                        tei_data, NID_PROGRAM_UID, orgunit_names,
                    )
                    patient_df = CSVIntegration.transform_events_to_patient_level(
                        events_df, NID_PROGRAM_UID,
                    )
                    if patient_df.empty:
                        continue
                    patient_df = self._fix_o2_data(events_df, patient_df)
                    patient_df = CSVIntegration.clean_transformed_dataframe(patient_df)
                    patient_df, removed_teis = preprocess_nid_enrollment_dates(patient_df)
                    if removed_teis:
                        logger.warning(f"  Dropped {len(removed_teis)} TEI(s) with invalid enrollment dates")

                    # Resolve region for this facility
                    facility_name = orgunit_names.get(ouid, ouid)
                    region_name = facility_to_region.get(facility_name)
                    region_uid = region_by_name.get(region_name) if region_name else None
                    patient_df["region_uid"] = region_uid or ""
                    patient_df["region_name"] = region_name or ""

                    # Save to regional file
                    if region_name:
                        regional_filename = regional_output_filename(region_name)
                        regional_target = os.path.join(NID_DIR, regional_filename)
                        append_missing_teis(patient_df, regional_target, merge_mode=self.merge_mode)

                    all_region_patient_dfs.append(patient_df)

                if not all_region_patient_dfs:
                    logger.warning("No NID data found for the specified org units")
                    return False
                combined_df = pd.concat(all_region_patient_dfs, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["tei_id"], keep="first")
                national_target = os.path.join(NID_DIR, NATIONAL_OUTPUT_FILE)
                append_missing_teis(combined_df, national_target, merge_mode=self.merge_mode)
                logger.info(f"Updated national file with {len(combined_df)} TEIs from specified org units")
            else:
                logger.info("Fetching region list...")
                regions = self.fetcher.fetch_all_regions()
                if not regions:
                    logger.error("No regions found")
                    return False

                total_regions = len(regions)
                for i, (region_uid, region_name) in enumerate(regions.items(), 1):
                    logger.info("-" * 80)
                    logger.info(f"[{i}/{total_regions}] Processing region: {region_name} ({region_uid})")
                    tei_data = self.fetcher.fetch_program_data(
                        NID_PROGRAM_UID, region_uid, "DESCENDANTS", 1000,
                        program_start_date=prog_start, program_end_date=prog_end,
                    )
                    tei_count = len(tei_data.get("trackedEntityInstances", []))
                    total_fetched_teis += tei_count
                    logger.info(f"Fetched TEIs from region scope: {tei_count}")
                    if tei_count == 0:
                        continue
                    events_df = CSVIntegration.create_events_dataframe(
                        tei_data, NID_PROGRAM_UID, orgunit_names,
                    )
                    patient_df = CSVIntegration.transform_events_to_patient_level(
                        events_df, NID_PROGRAM_UID,
                    )
                    if patient_df.empty:
                        logger.warning(f"No patient-level rows in {region_name}")
                        continue
                    patient_df = self._fix_o2_data(events_df, patient_df)
                    patient_df = CSVIntegration.clean_transformed_dataframe(patient_df)
                    patient_df, removed_teis = preprocess_nid_enrollment_dates(patient_df)
                    if removed_teis:
                        logger.warning(f"  {region_name}: dropped {len(removed_teis)} TEI(s) with invalid enrollment dates")
                    patient_df["region_uid"] = region_uid
                    patient_df["region_name"] = region_name
                    regional_filename = regional_output_filename(region_name)
                    regional_target = os.path.join(NID_DIR, regional_filename)
                    append_missing_teis(patient_df, regional_target, merge_mode=self.merge_mode)
                    all_region_patient_dfs.append(patient_df)

                if not all_region_patient_dfs:
                    logger.warning("No NID data found")
                    return False
                combined_df = pd.concat(all_region_patient_dfs, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["tei_id"], keep="first")
                national_target = os.path.join(NID_DIR, NATIONAL_OUTPUT_FILE)
                append_missing_teis(combined_df, national_target, merge_mode=self.merge_mode)

            logger.info("=" * 80)
            logger.info("NID FETCH COMPLETE")
            logger.info(f"Total TEIs fetched from DHIS2: {total_fetched_teis}")
            logger.info(f"Combined unique TEIs: {combined_df['tei_id'].nunique()}")
            logger.info(f"NID output directory: {NID_DIR}")
            logger.info(f"End time: {datetime.now()}")
            logger.info("=" * 80)
            return True

        except Exception as exc:
            logger.error(f"Learning facilities NID pipeline failed: {exc}")
            logger.error(traceback.format_exc())
            return False


def _prompt_facility_numbers(orgunit_names: dict) -> list:
    """Display numbered orgUnit list and return selected UIDs."""
    items = sorted(orgunit_names.items(), key=lambda x: x[1])
    print("\nAvailable facilities:")
    print("-" * 60)
    for idx, (uid, name) in enumerate(items, 1):
        print(f"  {idx:>4}. {name} ({uid})")
    print("-" * 60)
    raw = input("\nEnter facility numbers separated by comma (e.g., 1,3,5): ").strip()
    selected_indices = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            n = int(part)
            if 1 <= n <= len(items):
                selected_indices.add(n)
    return [items[i - 1][0] for i in sorted(selected_indices)]


def main() -> None:
    # Always prefer the repo .env for this command-line runner. This avoids
    # stale process or Windows environment variables shadowing recently edited
    # DHIS2 credentials.
    load_dotenv(os.path.join(REPO_ROOT, ".env"), override=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="",
        help="1/replace updates existing TEIs; 2/new_only appends only new TEIs.",
    )
    parser.add_argument(
        "--facilities",
        action="store_true",
        default=False,
        help="List all facilities, let user select by number, and fetch only those.",
    )
    args = parser.parse_args()

    mode_arg = args.mode.strip().lower()
    if mode_arg in {"1", "replace", "--replace"}:
        merge_mode = "replace"
    elif mode_arg in {"2", "new", "new_only", "--new-only"}:
        merge_mode = "new_only"
    else:
        choice = input(
            "Choose merge mode (1=replace existing TEIs, 2=fetch only new TEIs): "
        ).strip()
        merge_mode = "replace" if choice == "1" else "new_only"

    enrollment_date_filters = prompt_enrollment_date_filter()

    org_unit_ids = None
    if args.facilities:
        base_url = os.getenv("DHIS2_BASE_URL")
        username = os.getenv("DHIS2_USERNAME")
        password = os.getenv("DHIS2_PASSWORD")
        temp_fetcher = DHIS2DataFetcher(base_url, username, password)
        orgunit_names = temp_fetcher.fetch_orgunit_names()
        if not orgunit_names:
            print("No facilities found on DHIS2.")
            raise SystemExit(1)
        org_unit_ids = _prompt_facility_numbers(orgunit_names)
        if not org_unit_ids:
            print("No facilities selected.")
            raise SystemExit(0)

    pipeline = AutomatedLearningFacilitiesNIDPipeline(
        merge_mode=merge_mode,
        enrollment_date_filters=enrollment_date_filters,
        org_unit_ids=org_unit_ids,
    )
    success = pipeline.run_pipeline()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()

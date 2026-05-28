import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Set

import pandas as pd
from dotenv import load_dotenv

import dhis2_fetcher as core
from dhis2_fetcher import CSVIntegration, DHIS2DataFetcher, DEFAULT_OUTPUT_DIR, logger


NID_PROGRAM_UID = "pLk3Ht2XMKl"

ADDITIONAL_O2_DATAELEMENT_UID = "j4W59YyYG04"
ADDITIONAL_O2_DATAELEMENT_NAME = "Lowest recorded oxygen saturation (%)"
OBS_STAGE_UID = "VsVlpG1V2ub"  # Observations And Nursing Care 2

NID_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "nid")

LEARNING_FACILITIES_BY_REGION = {
    "afar": [],  # Empty list means include all facilities in this region
    "tigray": [
        "Adigudom primary hospital",
        "Hagereselam primary hospital",
        "Mekelle General Hospital",
        "Ayder referral hospital",
    ],
    "oromia": [
        "Adama Teaching Hospital",
        "Batu Primary Hospital",
        "Meki Primary Hospital",
        "Olenchity Primary Hospital",
    ],
    "amhara": [
        "Merawi PH",
        "Felegehiwot Referral",
        "Dangela PH",
        "Injibara General Hospital",
    ],
    "sidama": [
        "Adare GH",
        "Hawassa U/CSH",
        "Dore Bafano Primary hospital",
        "Tula Primary Hospital",
    ],
}

REGIONAL_OUTPUT_FILES = {
    "afar": "regional_Afar_newborn_nid.csv",
    "tigray": "regional_Tigray_newborn_nid.csv",
    "oromia": "regional_Oromia_newborn_nid.csv",
    "amhara": "regional_Amhara_newborn_nid.csv",
    "sidama": "regional_Sidama_newborn_nid.csv",
}

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
        ],
        "program_stage_name": "Admission Information",
    },
    "UK7jsDbVpj6": {
        "data_elements": ["CzIgD0rsk52"],
        "program_stage_name": "Maternal Birth And Infant Details",
    },
    "j0HI2eJjvbj": {
        "data_elements": ["gZi9y12E9i7"],
        "program_stage_name": "Observations And Nursing Care 1",
    },
    "ed8ErpgTCwx": {
        "data_elements": ["QK7Fi6OwtDC", "wlHEf9FdmJM", "sxtsEDilKZd"],
        "program_stage_name": "Interventions",
    },
    "TOicTEwzSGj": {
        "data_elements": ["vmOAGuFcaz4", "yBCwmQP0A6a", "wn0tHaHcceW"],
        "program_stage_name": "Discharge And Final Diagnosis",
    },
    "VsVlpG1V2ub": {
        "data_elements": ["nIKIu6f5vbW", ADDITIONAL_O2_DATAELEMENT_UID],
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
    "wn0tHaHcceW": "Sub-Categories of Infection",
    "A94ibeuO9GL": "Blood culture for suspected sepsis",
    "CzIgD0rsk52": "Birth weight (grams)",
    ADDITIONAL_O2_DATAELEMENT_UID: ADDITIONAL_O2_DATAELEMENT_NAME,
}


def normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("/", " ").split())


def ensure_additional_dataelement_mapping() -> None:
    """Force exact newborn mappings + additional oxygen saturation at runtime."""
    core.NEWBORN_PROGRAM_STAGE_MAPPING = EXPECTED_NEWBORN_STAGE_MAPPING
    core.DATA_ELEMENT_NAMES.update(EXPECTED_NEWBORN_DATAELEMENT_NAMES)
    for de_uid in EXPECTED_NEWBORN_DATAELEMENT_NAMES:
        core.NEWBORN_HEALTH_ELEMENTS.add(de_uid)
    core.REQUIRED_DATA_ELEMENTS = core.MATERNAL_HEALTH_ELEMENTS | core.NEWBORN_HEALTH_ELEMENTS
    logger.info("Applied expected newborn stage mapping and data element names (including oxygen saturation)")


def fetch_learning_facilities(fetcher: DHIS2DataFetcher) -> List[dict]:
    url = f"{fetcher.base_url}/api/organisationUnits.json"
    params = {
        "level": 3,
        "fields": "id,displayName,parent[id,name,displayName]",
        "paging": False,
    }
    resp = fetcher.session.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json().get("organisationUnits", [])


def build_learning_facility_ids_by_region(
    fetcher: DHIS2DataFetcher,
) -> Dict[str, Set[str]]:
    facilities = fetch_learning_facilities(fetcher)

    wanted = {
        region: {normalize(name) for name in names}
        for region, names in LEARNING_FACILITIES_BY_REGION.items()
    }

    region_facility_ids: Dict[str, Set[str]] = {k: set() for k in LEARNING_FACILITIES_BY_REGION}

    for ou in facilities:
        facility_name_norm = normalize(ou.get("displayName", ""))
        parent = ou.get("parent") or {}
        parent_name_norm = normalize(parent.get("displayName") or parent.get("name") or "")

        for region_key, names_norm in wanted.items():
            if parent_name_norm != region_key:
                continue

            # Empty list => include all facilities in this region
            if not names_norm:
                region_facility_ids[region_key].add(ou.get("id"))
            elif facility_name_norm in names_norm:
                region_facility_ids[region_key].add(ou.get("id"))

    for region_key, ids in region_facility_ids.items():
        configured_count = len(LEARNING_FACILITIES_BY_REGION[region_key])
        if configured_count == 0:
            logger.info(
                f"Facilities matched in {region_key.title()}: {len(ids)} (all region facilities included)"
            )
        else:
            logger.info(
                f"Learning facilities matched in {region_key.title()}: {len(ids)}/{configured_count}"
            )

    return region_facility_ids


def append_missing_teis(source_df: pd.DataFrame, target_path: str, merge_mode: str = "replace") -> int:
    source_df = source_df.copy()
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
        combined.to_csv(target_path, index=False, encoding="utf-8")

        mode_label = "Upserted" if merge_mode == "replace" else "Appended new"
        logger.info(
            f"{mode_label} {os.path.basename(target_path)}: "
            f"updated={updated_count}, added={len(to_add)}, total={len(combined)}"
        )
        return updated_count + len(to_add)

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
        self.fetcher = DHIS2DataFetcher(self.base_url, self.username, self.password)

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

            logger.info("Fetching region list...")
            regions = self.fetcher.fetch_all_regions()  # uid -> name
            if not regions:
                logger.error("No regions found")
                return False

            region_name_to_uid = {normalize(name): uid for uid, name in regions.items()}

            facility_ids_by_region = build_learning_facility_ids_by_region(self.fetcher)

            all_region_patient_dfs: List[pd.DataFrame] = []
            total_fetched_teis = 0

            for region_key, facility_ids in facility_ids_by_region.items():
                if not facility_ids:
                    logger.warning(f"Skipping {region_key.title()}: no matching learning facilities found")
                    continue

                region_uid = region_name_to_uid.get(region_key)
                if not region_uid:
                    logger.warning(f"Skipping {region_key.title()}: region UID not found")
                    continue

                logger.info("-" * 80)
                logger.info(f"Processing region: {region_key.title()} ({region_uid})")

                tei_data = self.fetcher.fetch_program_data(
                    NID_PROGRAM_UID,
                    region_uid,
                    "DESCENDANTS",
                    1000,
                )
                tei_count = len(tei_data.get("trackedEntityInstances", []))
                total_fetched_teis += tei_count
                logger.info(f"Fetched TEIs from region scope: {tei_count}")

                if tei_count == 0:
                    continue

                events_df = CSVIntegration.create_events_dataframe(
                    tei_data,
                    NID_PROGRAM_UID,
                    orgunit_names,
                )
                patient_df = CSVIntegration.transform_events_to_patient_level(
                    events_df,
                    NID_PROGRAM_UID,
                )

                if patient_df.empty:
                    logger.warning(f"No patient-level rows in {region_key.title()}")
                    continue

                patient_df = CSVIntegration.clean_transformed_dataframe(patient_df)

                if "orgUnit" not in patient_df.columns:
                    logger.warning("No orgUnit column in transformed data; cannot filter learning facilities")
                    continue

                before_filter = len(patient_df)
                patient_df = patient_df[
                    patient_df["orgUnit"].astype(str).str.strip().isin(facility_ids)
                ].copy()
                logger.info(
                    f"Filtered to learning facilities in {region_key.title()}: "
                    f"{len(patient_df)}/{before_filter} rows"
                )

                if patient_df.empty:
                    continue

                patient_df["region_uid"] = region_uid
                patient_df["region_name"] = regions[region_uid]

                regional_target = os.path.join(NID_DIR, REGIONAL_OUTPUT_FILES[region_key])
                append_missing_teis(patient_df, regional_target, merge_mode=self.merge_mode)

                all_region_patient_dfs.append(patient_df)

            if not all_region_patient_dfs:
                logger.warning("No learning facility data found to merge")
                return False

            combined_df = pd.concat(all_region_patient_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["tei_id"], keep="first")

            national_target = os.path.join(NID_DIR, NATIONAL_OUTPUT_FILE)
            append_missing_teis(combined_df, national_target, merge_mode=self.merge_mode)

            logger.info("=" * 80)
            logger.info("LEARNING FACILITIES NID FETCH + MERGE COMPLETE")
            logger.info(f"Total TEIs fetched from DHIS2 regional scopes: {total_fetched_teis}")
            logger.info(f"Combined unique TEIs from learning facilities: {combined_df['tei_id'].nunique()}")
            logger.info(f"NID output directory: {NID_DIR}")
            logger.info(f"End time: {datetime.now()}")
            logger.info("=" * 80)
            return True

        except Exception as exc:
            logger.error(f"Learning facilities NID pipeline failed: {exc}")
            logger.error(traceback.format_exc())
            return False


def main() -> None:
    load_dotenv()
    # Two choices:
    # 1 -> replace existing TEIs with freshly fetched values (upsert)
    # 2 -> fetch and append only new TEIs
    mode_arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ""
    if mode_arg in {"1", "replace", "--replace"}:
        merge_mode = "replace"
    elif mode_arg in {"2", "new", "new_only", "--new-only"}:
        merge_mode = "new_only"
    else:
        choice = input(
            "Choose merge mode (1=replace existing TEIs, 2=fetch only new TEIs): "
        ).strip()
        merge_mode = "replace" if choice == "1" else "new_only"

    pipeline = AutomatedLearningFacilitiesNIDPipeline(merge_mode=merge_mode)
    success = pipeline.run_pipeline()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()

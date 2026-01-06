import pandas as pd
import numpy as np
import requests
import json
import os
import sys
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Any, Union
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
from queue import Queue
import time
import traceback
import webbrowser
import re
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
# Program UIDs
MATERNAL_PROGRAM_UID = "aLoraiFNkng"
NEWBORN_PROGRAM_UID = "pLk3Ht2XMKl"  # UPDATED: Your specified newborn UID

# FIX 1: Correct path calculation for your project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_UTILS_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "imnid")
DEFAULT_MATERNAL_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "maternal")
DEFAULT_NEWBORN_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "newborn")


# Define data element mappings - EXACT SAME AS data_service.py
MATERNAL_HEALTH_ELEMENTS = {
    "Q1p7CxWGUoi",
    "lphtwP2ViZU",
    "wZig9cek3Gv",
    "VzwnSBROvUm",
    "tIa0WvbPGLk",
    "z7Eb2yFLOBI",
    "TjQOcW6tm8k",
    "CJiTafFo0TS",
    "yVRLuRU943e",
    "tTrH9cOQRnZ",
    "H7J2SxBpObS",
    "QUlJEvzGcQK",
    "K8BCYRU1TUP",
}

NEWBORN_HEALTH_ELEMENTS = {
    "QK7Fi6OwtDC",
    "yxWUMt3sCil",
    "gZi9y12E9i7",
    "UOmhJkyAK6h",
    "wlHEf9FdmJM",
    "T30GbTiVgFR",
    "OpHw2X58x5i",
    "gJH6PkYI6IV",
    "aK5txmRYpVX",
    "vmOAGuFcaz4",
    "yBCwmQP0A6a",
    "nIKIu6f5vbW",
    "sxtsEDilKZd",
    "wn0tHaHcceW",
    "A94ibeuO9GL",
    "CzIgD0rsk52",  # ADDED: Birth weight
}

REQUIRED_DATA_ELEMENTS = MATERNAL_HEALTH_ELEMENTS | NEWBORN_HEALTH_ELEMENTS

# EXACT SAME DATA_ELEMENT_NAMES as in data_service.py - UPDATED WITH BIRTH WEIGHT
DATA_ELEMENT_NAMES = {
    # Maternal Health
    "Q1p7CxWGUoi": "FP Counseling and Method Provided pp",
    "lphtwP2ViZU": "Mode of Delivery maternal",
    "wZig9cek3Gv": "Birth Outcome",
    "VzwnSBROvUm": "Number of Newborns",
    "tIa0WvbPGLk": "Other Number of Newborns",
    "z7Eb2yFLOBI": "Date/stay pp",
    "TjQOcW6tm8k": "Condition of Discharge",
    "CJiTafFo0TS": "Obstetric condition at delivery",
    "yVRLuRU943e": "Uterotonics given",
    "tTrH9cOQRnZ": "HIV Result",
    "H7J2SxBpObS": "ARV Rx for Newborn (By type) pp",
    "QUlJEvzGcQK": "Birth Weight (grams)",
    "K8BCYRU1TUP": "Instrumental delivery",
    # Newborn Health
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
    "CzIgD0rsk52": "Birth weight (grams)",  # ADDED: Birth weight
}

# EXACT SAME PROGRAM STAGE MAPPINGS as in data_service.py - WITH HUMAN-READABLE NAMES
MATERNAL_PROGRAM_STAGE_MAPPING = {
    "mdw5BoS50mb": {  # Delivery summary
        "data_elements": [
            "lphtwP2ViZU",
            "VzwnSBROvUm",
            "QUlJEvzGcQK",
            "tTrH9cOQRnZ",
            "wZig9cek3Gv",
            "tIa0WvbPGLk",
            "CJiTafFo0TS",
            "yVRLuRU943e",
        ],
        "program_stage_name": "Delivery summary",
    },
    "VpBHRE7FlJL": {  # Postpartum care
        "data_elements": ["z7Eb2yFLOBI", "H7J2SxBpObS", "Q1p7CxWGUoi"],
        "program_stage_name": "Postpartum care",
    },
    "DLVsIxjhwMj": {  # Discharge Summary
        "data_elements": ["TjQOcW6tm8k"],
        "program_stage_name": "Discharge Summary",
    },
    "bwk9bBfYcsD": {  # Instrumental Delivery form
        "data_elements": ["K8BCYRU1TUP"],
        "program_stage_name": "Instrumental Delivery form",
    },
}

# UPDATED NEWBORN PROGRAM STAGE MAPPING - Added birth weight and corrected UID
NEWBORN_PROGRAM_STAGE_MAPPING = {
    "l39SlVGlQGs": {  # Admission Information
        "data_elements": [
            "UOmhJkyAK6h",  # Date of Admission
            "yxWUMt3sCil",  # Weight on admission
            "T30GbTiVgFR",  # First Reason for Admission
            "OpHw2X58x5i",  # Second Reason for Admission
            "gJH6PkYI6IV",  # Third Reason for Admission
            "aK5txmRYpVX",  # birth location (inborn/outborn)
        ],
        "program_stage_name": "Admission Information",
    },
    "UK7jsDbVpj6": {  # Maternal Birth And Infant Details - UPDATED UID
        "data_elements": [
            "CzIgD0rsk52",  # Birth weight (grams) - ADDED
        ],
        "program_stage_name": "Maternal Birth And Infant Details",
    },
    "j0HI2eJjvbj": {  # Observations And Nursing Care 1
        "data_elements": [
            "gZi9y12E9i7",  # Temperature on admission
        ],
        "program_stage_name": "Observations And Nursing Care 1",
    },
    "ed8ErpgTCwx": {  # Interventions
        "data_elements": [
            "QK7Fi6OwtDC",  # KMC Administered
            "wlHEf9FdmJM",  # CPAP Administered
            "sxtsEDilKZd",  # Were antibiotics administered?
        ],
        "program_stage_name": "Interventions",
    },
    "TOicTEwzSGj": {  # Discharge And Final Diagnosis
        "data_elements": [
            "vmOAGuFcaz4",  # Newborn status at discharge
            "yBCwmQP0A6a",  # Discharge Weight (grams):
            "wn0tHaHcceW",  # Sub-Categories of Infection:
        ],
        "program_stage_name": "Discharge And Final Diagnosis",
    },
    "VsVlpG1V2ub": {  # Observations And Nursing Care 2
        "data_elements": [
            "nIKIu6f5vbW",  # lowest recorded temperature (Celsius)
        ],
        "program_stage_name": "Observations And Nursing Care 2",
    },
    "aCrttmnx7FI": {  # Microbiology And Labs
        "data_elements": [
            "A94ibeuO9GL",  # Blood culture for suspected sepsis:
        ],
        "program_stage_name": "Microbiology And Labs",
    },
}


class DHIS2DataFetcher:
    def __init__(self, base_url: str, username: str, password: str):
        """
        Initialize DHIS2 connection
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )
        self.session.timeout = 300

    def fetch_all_regions(self) -> Dict[str, str]:
        """Fetch all regions from DHIS2 (level=2 org units)"""
        url = f"{self.base_url}/api/organisationUnits.json"
        params = {"level": 2, "fields": "id,displayName", "paging": False}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            regions = {}
            for ou in data.get("organisationUnits", []):
                regions[ou["id"]] = ou["displayName"]

            return regions
        except Exception as e:
            logger.error(f"Failed to fetch regions: {e}")
            return {}

    def fetch_program_data(
        self,
        program_uid: str,
        org_unit_uid: str,
        org_unit_mode: str = "DESCENDANTS",
        page_size: int = 1000,
    ) -> Dict:
        """
        Fetch TEI data for a specific program and org unit with pagination
        FIXED: Better pagination handling to ensure all TEIs are fetched
        """
        url = f"{self.base_url}/api/trackedEntityInstances.json"

        params = {
            "program": program_uid,
            "ou": org_unit_uid,
            "ouMode": org_unit_mode,
            "fields": "trackedEntityInstance,orgUnit,attributes[attribute,value],"
            "enrollments[enrollment,enrollmentDate,incidentDate,status,orgUnit,"
            "events[event,programStage,eventDate,orgUnit,"
            "dataValues[dataElement,value]]]",
            "pageSize": page_size,
            "totalPages": True,
        }

        logger.info(f"Fetching data for program {program_uid}, org unit {org_unit_uid}")

        all_teis = []
        page = 1
        total_items = 0

        try:
            while True:
                params["page"] = page
                logger.info(f"📄 Fetching page {page}...")
                response = self.session.get(url, params=params, timeout=300)
                response.raise_for_status()
                data = response.json()

                teis = data.get("trackedEntityInstances", [])
                tei_count = len(teis)
                all_teis.extend(teis)

                # Get pagination info
                pager = data.get("pager", {})

                # Check if we have a pager object
                if pager and "pageCount" in pager:
                    total_pages = pager.get("pageCount", 1)
                    current_page = pager.get("page", page)
                    total_items = pager.get("total", len(all_teis))

                    logger.info(
                        f"Page {current_page}/{total_pages}: {tei_count} TEIs (Total so far: {len(all_teis)}/{total_items})"
                    )

                    # Check if we've reached the last page
                    if current_page >= total_pages or not teis:
                        logger.info(
                            f"✅ Completed pagination. Total TEIs fetched: {len(all_teis)}"
                        )
                        break

                    page += 1
                else:
                    # No pager means single page or all data
                    logger.info(f"Single page: {tei_count} TEIs")
                    break

                # Small delay to be nice to the server
                time.sleep(0.3)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text}")
            # Return whatever we've fetched so far
            return {"trackedEntityInstances": all_teis}

        # Additional verification
        logger.info(f"📊 Verification: Fetched {len(all_teis)} TEIs")

        # Check for duplicates
        tei_ids = [
            tei.get("trackedEntityInstance")
            for tei in all_teis
            if tei.get("trackedEntityInstance")
        ]
        unique_tei_ids = set(tei_ids)
        if len(tei_ids) != len(unique_tei_ids):
            logger.warning(
                f"⚠️ Found {len(tei_ids) - len(unique_tei_ids)} duplicate TEIs"
            )
            # Remove duplicates while preserving order
            seen = set()
            unique_teis = []
            for tei in all_teis:
                tei_id = tei.get("trackedEntityInstance")
                if tei_id and tei_id not in seen:
                    seen.add(tei_id)
                    unique_teis.append(tei)
            all_teis = unique_teis
            logger.info(f"🔄 After deduplication: {len(all_teis)} unique TEIs")

        return {"trackedEntityInstances": all_teis}

    def fetch_orgunit_names(self) -> Dict[str, str]:
        """Fetch all orgUnit names"""
        url = f"{self.base_url}/api/organisationUnits.json"
        params = {"fields": "id,displayName", "paging": False}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            ou_dict = {}
            for ou in data.get("organisationUnits", []):
                ou_dict[ou["id"]] = ou["displayName"]

            return ou_dict
        except Exception as e:
            logger.error(f"Failed to fetch orgUnit names: {e}")
            return {}

    def fetch_facility_to_region_mapping(self) -> Dict[str, str]:
        """
        Fetch facility to region mapping (facility name -> region name)
        This helps map CSV facilities to their correct regions
        """
        url = f"{self.base_url}/api/organisationUnits.json"
        params = {
            "level": 3,  # Facilities are usually level 3
            "fields": "id,displayName,parent[name]",
            "paging": False,
        }

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            facility_to_region = {}
            for ou in data.get("organisationUnits", []):
                facility_name = ou["displayName"]
                if "parent" in ou and "name" in ou["parent"]:
                    region_name = ou["parent"]["name"]
                    facility_to_region[facility_name] = region_name

            logger.info(
                f"Fetched facility-to-region mapping for {len(facility_to_region)} facilities"
            )
            return facility_to_region
        except Exception as e:
            logger.error(f"Failed to fetch facility-to-region mapping: {e}")
            return {}


class CSVIntegration:
    """Class to handle CSV integration similar to data_service.py"""

    @staticmethod
    def get_facility_region_mapping(
        csv_df: pd.DataFrame, facility_to_region_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Add region information to CSV data based on facility names
        Returns CSV with 'csv_region' column added
        """
        if csv_df.empty or "orgUnit_name" not in csv_df.columns:
            return csv_df

        df = csv_df.copy()

        # Initialize csv_region column
        df["csv_region"] = ""

        # Map each facility to its region
        for idx, row in df.iterrows():
            facility_name = row.get("orgUnit_name", "")
            if facility_name and facility_name in facility_to_region_map:
                df.at[idx, "csv_region"] = facility_to_region_map[facility_name]
            else:
                # Try to match partial names
                for facility_key, region_name in facility_to_region_map.items():
                    if facility_key in facility_name or facility_name in facility_key:
                        df.at[idx, "csv_region"] = region_name
                        break

        # Log mapping statistics
        mapped_count = df[df["csv_region"] != ""].shape[0]
        total_count = df.shape[0]
        logger.info(
            f"📊 CSV region mapping: {mapped_count}/{total_count} rows mapped ({mapped_count/total_count*100:.1f}%)"
        )

        return df

    @staticmethod
    def filter_csv_data_by_user_access(
        csv_df: pd.DataFrame,
        user_level: str,
        user_region: str = None,
        facility_to_region_map: Dict[str, str] = None,
    ) -> pd.DataFrame:
        """
        Filter CSV data based on user's access level - FIXED LOGIC
        """
        if csv_df.empty:
            return csv_df

        # Check if CSV has orgUnit_name column
        if "orgUnit_name" not in csv_df.columns:
            logger.warning(
                "CSV data does not have 'orgUnit_name' column - cannot filter by facility"
            )
            return csv_df

        if user_level == "national":
            # National users see ALL CSV data
            logger.info("🌍 National user - returning ALL CSV data")
            return csv_df

        elif user_level == "regional":
            # Regional users see only facilities in their region
            if not user_region:
                logger.warning(
                    "Regional user has no region_name - cannot filter CSV data"
                )
                return pd.DataFrame()

            # If we have facility-to-region mapping, use it
            if facility_to_region_map:
                logger.info(
                    f"🏞️ Regional user '{user_region}' - filtering CSV by facility-to-region mapping"
                )

                # Add region column if not present
                if "csv_region" not in csv_df.columns:
                    csv_df = CSVIntegration.get_facility_region_mapping(
                        csv_df, facility_to_region_map
                    )

                # Filter by region
                filtered_csv = csv_df[csv_df["csv_region"] == user_region]

                logger.info(
                    f"   Filtered CSV: {len(filtered_csv)}/{len(csv_df)} rows for region '{user_region}'"
                )
                return filtered_csv
            else:
                # Fallback: filter by orgUnit_name (assuming orgUnit_name is the region)
                filtered_csv = csv_df[csv_df["orgUnit_name"] == user_region]
                logger.info(
                    f"🏞️ Regional user '{user_region}' - filtered CSV to {len(filtered_csv)} rows (using orgUnit_name)"
                )
                return filtered_csv

        else:
            logger.warning(f"Unknown user level '{user_level}' - returning no CSV data")
            return pd.DataFrame()

    @staticmethod
    def integrate_maternal_csv_data_for_region(
        evt_df: pd.DataFrame, csv_df: pd.DataFrame, target_region: str = None
    ) -> pd.DataFrame:
        """
        Smart integration of maternal CSV data for a SPECIFIC REGION
        Only integrates CSV data that belongs to the target region
        """
        if evt_df.empty or csv_df.empty:
            return evt_df

        # Filter CSV data for this specific region
        if target_region and "csv_region" in csv_df.columns:
            region_csv_df = csv_df[csv_df["csv_region"] == target_region].copy()
            logger.info(
                f"📁 Integrating CSV data for region '{target_region}': {len(region_csv_df)} rows"
            )
        else:
            # If no region filter or no region column, use all CSV data
            region_csv_df = csv_df.copy()
            logger.info(
                f"📁 Integrating ALL CSV data (no region filter): {len(region_csv_df)} rows"
            )

        if region_csv_df.empty:
            logger.info(
                f"⚠️ No CSV data for region '{target_region}' - using DHIS2 data only"
            )
            return evt_df

        # Get unique TEI IDs from both sources
        csv_tei_ids = (
            set(region_csv_df["tei_id"].unique())
            if "tei_id" in region_csv_df.columns
            else set()
        )
        dhis2_tei_ids = (
            set(evt_df["tei_id"].unique()) if "tei_id" in evt_df.columns else set()
        )

        logger.info(f"📊 TEI Statistics for region {target_region}:")
        logger.info(f"   DHIS2: {len(dhis2_tei_ids)} TEIs")
        logger.info(f"   CSV: {len(csv_tei_ids)} TEIs")

        # Identify placeholder vs actual events in both sources
        if "has_actual_event" not in region_csv_df.columns:
            region_csv_df["has_actual_event"] = True

        # STRATEGY 1: Handle TEIs that exist in both sources
        common_teis = csv_tei_ids.intersection(dhis2_tei_ids)
        teis_to_remove_from_dhis2 = set()

        for tei_id in common_teis:
            # Get events for this TEI from both sources
            dhis2_tei_events = evt_df[evt_df["tei_id"] == tei_id]
            csv_tei_events = region_csv_df[region_csv_df["tei_id"] == tei_id]

            # Check if DHIS2 has ANY placeholder events for this TEI
            dhis2_has_placeholders = any(dhis2_tei_events["has_actual_event"] == False)
            dhis2_has_actual_events = any(dhis2_tei_events["has_actual_event"] == True)

            # Check if CSV has placeholder events for this TEI
            csv_has_placeholders = any(csv_tei_events["has_actual_event"] == False)
            csv_has_actual_events = any(csv_tei_events["has_actual_event"] == True)

            # STRATEGY 1A: Both have placeholders → prioritize CSV
            if dhis2_has_placeholders and csv_has_placeholders:
                logger.info(
                    f"🔄 TEI {tei_id}: Both sources have placeholders - prioritizing CSV data"
                )
                teis_to_remove_from_dhis2.add(tei_id)

            # STRATEGY 1B: DHIS2 has placeholders, CSV has actual events → replace with CSV
            elif dhis2_has_placeholders and csv_has_actual_events:
                logger.info(
                    f"🔄 TEI {tei_id}: Replacing DHIS2 placeholder with CSV actual events"
                )
                teis_to_remove_from_dhis2.add(tei_id)

            # STRATEGY 1C: Both have actual events → handle event by event
            elif dhis2_has_actual_events and csv_has_actual_events:
                # Get event IDs from both sources
                dhis2_event_ids = set(dhis2_tei_events["event"].unique())
                csv_event_ids = set(csv_tei_events["event"].unique())

                # Find overlapping event IDs (same events in both sources)
                overlapping_events = dhis2_event_ids.intersection(csv_event_ids)

                if overlapping_events:
                    logger.info(
                        f"🔄 TEI {tei_id}: Replacing {len(overlapping_events)} overlapping events"
                    )
                    # Remove DHIS2 events that overlap with CSV events
                    evt_df = evt_df[
                        ~(
                            (evt_df["tei_id"] == tei_id)
                            & (evt_df["event"].isin(overlapping_events))
                        )
                    ]

                # Note: Events with different IDs will be kept in both sources
                new_events_in_csv = csv_event_ids - dhis2_event_ids
                if new_events_in_csv:
                    logger.info(
                        f"📥 TEI {tei_id}: Adding {len(new_events_in_csv)} new events from CSV"
                    )

            # STRATEGY 1D: DHIS2 has actual events, CSV has placeholders → unusual case, log it
            elif dhis2_has_actual_events and csv_has_placeholders:
                logger.warning(
                    f"⚠️ TEI {tei_id}: DHIS2 has actual events but CSV has placeholders - keeping both"
                )

        # Remove TEIs that should be completely replaced by CSV
        if teis_to_remove_from_dhis2:
            logger.info(
                f"🗑️ Removing {len(teis_to_remove_from_dhis2)} TEIs from DHIS2 (replaced by CSV)"
            )
            evt_df = evt_df[~evt_df["tei_id"].isin(teis_to_remove_from_dhis2)]

        # STRATEGY 2: Add completely new TEIs from CSV (not in DHIS2 at all)
        new_teis_in_csv = csv_tei_ids - dhis2_tei_ids
        if new_teis_in_csv:
            logger.info(
                f"🆕 Adding {len(new_teis_in_csv)} completely new TEIs from CSV for region {target_region}"
            )

        # Add all CSV data for this region
        original_count = len(evt_df)
        evt_df = pd.concat([evt_df, region_csv_df], ignore_index=True)
        final_count = len(evt_df)

        logger.info(f"✅ REGION {target_region} INTEGRATION COMPLETE:")
        logger.info(f"   📈 Before integration: {original_count} rows")
        logger.info(f"   📈 After integration: {final_count} rows")
        logger.info(f"   📈 Net change: {final_count - original_count} rows")
        logger.info(f"   👥 Unique TEIs: {len(evt_df['tei_id'].unique())}")

        return evt_df

    @staticmethod
    def clean_column_name(name: str) -> str:
        """
        Clean a name to be used as a column name.
        Removes special characters, replaces spaces with underscores, etc.
        Returns lowercase string.
        """
        if not isinstance(name, str):
            name = str(name)

        # Convert to lowercase first
        name = name.lower()

        # Replace problematic characters
        cleaned = (
            name.replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(".", "")
            .replace(",", "")
            .replace("-", "_")
            .replace("'", "")
            .replace('"', "")
            .replace("&", "and")
            .replace("%", "pct")
            .replace("#", "num")
            .replace("°", "deg")
            .replace("+", "plus")
            .replace("=", "equals")
            .replace("<", "lt")
            .replace(">", "gt")
        )

        # Remove multiple underscores
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")

        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")

        # If empty after cleaning, return "unknown"
        if not cleaned:
            return "unknown"

        return cleaned

    @staticmethod
    def create_events_dataframe(
        tei_data: Dict, program_uid: str, orgunit_names: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Create events DataFrame from TEI data with placeholders
        Uses ONLY human-readable program stage names
        ALL dates use original DHIS2 format
        NO period, period_display, or period_sort columns
        """
        events_list = []

        # Determine program type and mapping
        is_maternal = program_uid == MATERNAL_PROGRAM_UID
        if is_maternal:
            PROGRAM_STAGE_MAPPING = MATERNAL_PROGRAM_STAGE_MAPPING
            logger.info("📋 Using MATERNAL program stage mapping")
        else:
            PROGRAM_STAGE_MAPPING = NEWBORN_PROGRAM_STAGE_MAPPING
            logger.info("📋 Using NEWBORN program stage mapping")

        # Track all TEI IDs
        all_tei_ids = {
            tei.get("trackedEntityInstance")
            for tei in tei_data.get("trackedEntityInstances", [])
        }

        # Track which TEI-ProgramStage combinations have actual events
        tei_program_stage_events = set()

        # Track enrollment information for each TEI
        enrollment_info = (
            {}
        )  # tei_id -> {enrollment_status, enrollment_date, incident_date}

        # Process actual events from DHIS2
        for tei in tei_data.get("trackedEntityInstances", []):
            tei_id = tei.get("trackedEntityInstance")
            tei_org_unit = tei.get("orgUnit")

            # Store enrollment information from the first enrollment
            enrollment_status = ""
            enrollment_date = ""
            incident_date = ""

            for enrollment in tei.get("enrollments", []):
                # Extract enrollment information (from first enrollment found)
                enrollment_status = enrollment.get("status", "")
                enrollment_date = enrollment.get("enrollmentDate", "")
                incident_date = enrollment.get("incidentDate", "")

                # Store enrollment info for this TEI
                enrollment_info[tei_id] = {
                    "enrollment_status": enrollment_status,
                    "enrollment_date": enrollment_date,
                    "incident_date": incident_date,
                }

                for event in enrollment.get("events", []):
                    event_orgUnit = (
                        event.get("orgUnit")
                        or enrollment.get("orgUnit")
                        or tei_org_unit
                    )
                    event_date = event.get("eventDate")
                    program_stage_uid = event.get("programStage")

                    # Mark that this TEI has an event for this program stage UID
                    tei_program_stage_events.add((tei_id, program_stage_uid))

                    # CRITICAL FIX: Get program stage name from OUR MAPPING, not from DHIS2
                    # This ensures consistent human-readable names
                    program_stage_info = PROGRAM_STAGE_MAPPING.get(
                        program_stage_uid, {}
                    )
                    program_stage_name = program_stage_info.get(
                        "program_stage_name",
                        f"Unknown_{program_stage_uid}",  # Fallback if not in mapping
                    )

                    # Get all required data elements for this program stage
                    stage_required_elements = program_stage_info.get(
                        "data_elements", []
                    )

                    # Process ALL required data elements for this program stage
                    for data_element_uid in stage_required_elements:
                        # Find the data value for this element (if it exists)
                        data_value = ""
                        for dv in event.get("dataValues", []):
                            if dv.get("dataElement") == data_element_uid:
                                data_value = dv.get("value", "")
                                break

                        # UPDATE: Check if this is ARV Rx for Newborn (By type) pp element
                        if data_element_uid == "H7J2SxBpObS":
                            # Convert "yes" to "N/A"
                            if (
                                isinstance(data_value, str)
                                and data_value.lower().strip() == "yes"
                            ):
                                data_value = "N/A"
                                logger.debug(
                                    f"Updated ARV Rx value from 'yes' to 'N/A' for TEI {tei_id}"
                                )

                        data_element_name = DATA_ELEMENT_NAMES.get(
                            data_element_uid,
                            data_element_uid,  # Fallback to UID if not in mapping
                        )

                        event_data = {
                            "tei_id": tei_id,
                            "event": event.get("event"),
                            "programStage_uid": program_stage_uid,
                            "programStageName": program_stage_name,  # HUMAN-READABLE NAME
                            "orgUnit": event_orgUnit,
                            "orgUnit_name": orgunit_names.get(
                                event_orgUnit, "Unknown OrgUnit"
                            ),
                            "eventDate": event_date,  # Keep original DHIS2 format
                            "dataElement_uid": data_element_uid,
                            "dataElementName": data_element_name,
                            "value": data_value,
                            "has_actual_event": True,
                            # Add enrollment information - keep original DHIS2 format
                            "enrollment_status": enrollment_status,
                            "enrollment_date": enrollment_date,  # Keep original DHIS2 format
                            "incident_date": incident_date,  # Keep original DHIS2 format
                        }
                        events_list.append(event_data)

        # Create placeholder events for TEI-ProgramStage combinations that have NO events
        placeholder_count = 0
        for tei_id in all_tei_ids:
            # Find tei info for enrollment date and enrollment info
            tei_info = {}
            for tei in tei_data.get("trackedEntityInstances", []):
                if tei.get("trackedEntityInstance") == tei_id:
                    tei_info = {
                        "orgUnit": tei.get("orgUnit"),
                        "enrollmentDate": (
                            tei.get("enrollments", [{}])[0].get("enrollmentDate")
                            if tei.get("enrollments")
                            else None
                        ),
                    }
                    break

            # Get enrollment information for this TEI (if available)
            tei_enrollment_info = enrollment_info.get(tei_id, {})
            enrollment_status = tei_enrollment_info.get("enrollment_status", "")
            enrollment_date = tei_enrollment_info.get("enrollment_date", "")
            incident_date = tei_enrollment_info.get("incident_date", "")

            # Check each required program stage by UID
            for program_stage_uid, stage_info in PROGRAM_STAGE_MAPPING.items():
                # Check if this TEI has ANY event for this program stage UID
                has_event_for_stage = (
                    tei_id,
                    program_stage_uid,
                ) in tei_program_stage_events

                if not has_event_for_stage:
                    # This TEI has NO events for this program stage - create placeholders
                    event_date = tei_info.get("enrollmentDate")
                    program_stage_name = stage_info[
                        "program_stage_name"
                    ]  # HUMAN-READABLE NAME

                    # Create one placeholder row for EACH data element in this program stage
                    for data_element_uid in stage_info["data_elements"]:
                        data_element_name = DATA_ELEMENT_NAMES.get(
                            data_element_uid,
                            data_element_uid,
                        )

                        placeholder_event = {
                            "tei_id": tei_id,
                            "event": f"placeholder_{tei_id}_{program_stage_uid}",
                            "programStage_uid": program_stage_uid,
                            "programStageName": program_stage_name,  # HUMAN-READABLE NAME
                            "orgUnit": tei_info.get("orgUnit"),
                            "orgUnit_name": orgunit_names.get(
                                tei_info.get("orgUnit"), "Unknown OrgUnit"
                            ),
                            "eventDate": event_date,  # Keep original format
                            "dataElement_uid": data_element_uid,
                            "dataElementName": data_element_name,
                            "value": "",  # Empty value for missing data
                            "has_actual_event": False,
                            # Add enrollment information (even for placeholders) - keep original format
                            "enrollment_status": enrollment_status,
                            "enrollment_date": enrollment_date,  # Keep original DHIS2 format
                            "incident_date": incident_date,  # Keep original DHIS2 format
                        }
                        events_list.append(placeholder_event)
                        placeholder_count += 1

        evt_df = pd.DataFrame(events_list)

        # Log program stage names being used (for verification)
        unique_stages = evt_df["programStageName"].unique()
        logger.info(f"📊 Program stage names in data ({len(unique_stages)} unique):")
        for stage in unique_stages:
            logger.info(f"   - {stage}")

        # Log enrollment information availability
        enrollment_info_count = evt_df[evt_df["enrollment_status"] != ""].shape[0]
        logger.info(
            f"📋 Enrollment information available for {enrollment_info_count} rows"
        )

        # Log date formatting
        if "eventDate" in evt_df.columns and not evt_df.empty:
            sample_dates = evt_df["eventDate"].dropna().unique()[:5]
            logger.info(f"📅 Sample event dates (original DHIS2 format):")
            for date_str in sample_dates:
                logger.info(f"   - {date_str}")

        if "enrollment_date" in evt_df.columns and not evt_df.empty:
            sample_dates = evt_df["enrollment_date"].dropna().unique()[:5]
            logger.info(f"📅 Sample enrollment dates (original DHIS2 format):")
            for date_str in sample_dates:
                logger.info(f"   - {date_str}")

        if "incident_date" in evt_df.columns and not evt_df.empty:
            sample_dates = evt_df["incident_date"].dropna().unique()[:5]
            logger.info(f"📅 Sample incident dates (original DHIS2 format):")
            for date_str in sample_dates:
                logger.info(f"   - {date_str}")

        logger.info(f"Created {len(evt_df)} events ({placeholder_count} placeholders)")

        return evt_df

    @staticmethod
    def transform_events_to_patient_level(
        events_df: pd.DataFrame, program_uid: str
    ) -> pd.DataFrame:
        """
        Transform events DataFrame to patient-level format
        UPDATED: For newborn program, DO NOT create versioned columns since we don't have repeatable stages
        For maternal program, keep versioned columns for repeatable stages
        """
        if events_df.empty:
            logger.warning("Empty events DataFrame provided for transformation")
            return pd.DataFrame()

        # Make a copy to avoid modifying the original
        df = events_df.copy()

        # Determine program type based on UID
        is_maternal = program_uid == MATERNAL_PROGRAM_UID
        program_type = "maternal" if is_maternal else "newborn"

        logger.info(
            f"🔄 Transforming {program_type.upper()} events to patient-level format"
        )
        logger.info(
            f"   Input: {len(df)} rows from {df['tei_id'].nunique()} unique patients"
        )

        # Check what program stage names we have
        unique_stages = df["programStageName"].unique()
        logger.info(f"   Found {len(unique_stages)} unique program stages")

        # Step 1: Create base patient information including enrollment data
        patient_base = df[["tei_id"]].drop_duplicates()

        # Add orgUnit and orgUnit_name if available
        if "orgUnit" in df.columns:
            org_mapping = (
                df[["tei_id", "orgUnit"]]
                .drop_duplicates()
                .set_index("tei_id")["orgUnit"]
                .to_dict()
            )
            patient_base["orgUnit"] = patient_base["tei_id"].map(org_mapping)

        if "orgUnit_name" in df.columns:
            org_name_mapping = (
                df[["tei_id", "orgUnit_name"]]
                .drop_duplicates()
                .set_index("tei_id")["orgUnit_name"]
                .to_dict()
            )
            patient_base["orgUnit_name"] = patient_base["tei_id"].map(org_name_mapping)

        # Add enrollment information to base patient data - use original dates
        enrollment_cols = ["enrollment_status", "enrollment_date", "incident_date"]
        for col in enrollment_cols:
            if col in df.columns:
                # Get the first non-empty value for each TEI
                col_mapping = (
                    df[["tei_id", col]]
                    .replace("", np.nan)  # Replace empty strings with NaN
                    .dropna()  # Drop rows where this column is NaN
                    .drop_duplicates(
                        subset=["tei_id"], keep="first"
                    )  # Keep first non-empty value
                    .set_index("tei_id")[col]
                    .to_dict()
                )
                patient_base[col] = patient_base["tei_id"].map(col_mapping)

        # Step 2: Get all unique program stages for this data
        program_stages = df[["programStageName"]].drop_duplicates()

        # Step 3: Process each program stage separately
        all_stage_data = []

        for _, stage_row in program_stages.iterrows():
            program_stage_name = stage_row["programStageName"]

            # Filter data for this program stage
            stage_data = df[df["programStageName"] == program_stage_name].copy()

            if stage_data.empty:
                continue

            logger.info(f"   Processing program stage: {program_stage_name}")

            # Group by tei_id - each group represents one patient in this program stage
            stage_groups = stage_data.groupby("tei_id")

            stage_patient_rows = []

            for tei_id, group in stage_groups:
                # Start with basic patient info
                patient_row = {"tei_id": tei_id}

                # Clean program stage name for column naming
                clean_stage_name = CSVIntegration.clean_column_name(program_stage_name)

                # CRITICAL CHANGE: For NEWBORN program, DO NOT create versioned columns
                # For MATERNAL program, create versioned columns for repeatable stages
                if not is_maternal:
                    # NEWBORN PROGRAM: No version suffixes (non-repeatable stages)
                    # Find the actual event (skip placeholders)
                    actual_events = group[
                        ~group["event"].astype(str).str.startswith("placeholder_")
                    ]

                    if not actual_events.empty:
                        # Use the first actual event found
                        first_event_row = actual_events.iloc[0]

                        # Add event metadata WITHOUT version suffix
                        if "event" in first_event_row:
                            patient_row[f"event_{clean_stage_name}"] = first_event_row[
                                "event"
                            ]

                        # Use eventDate as-is (original DHIS2 format) WITHOUT version suffix
                        if "eventDate" in first_event_row:
                            patient_row[f"event_date_{clean_stage_name}"] = (
                                first_event_row.get("eventDate", "")
                            )

                        # Process all data elements for this specific event WITHOUT version suffix
                        for idx, row in actual_events.iterrows():
                            data_element_name = row["dataElementName"]
                            value = row["value"]
                            data_element_uid = row.get("dataElement_uid", "")

                            # Skip empty values
                            if pd.isna(value) or str(value).strip() == "":
                                continue

                            # Clean data element name
                            clean_element_name = CSVIntegration.clean_column_name(
                                data_element_name
                            )

                            # Create column name WITHOUT version suffix for newborn
                            column_name = f"{clean_element_name}_{clean_stage_name}"

                            # Add to patient row
                            patient_row[column_name] = value
                else:
                    # MATERNAL PROGRAM: Original logic with versioned columns for repeatable stages
                    # FIRST: Group events by their event ID (each event has its own event ID and event date)
                    # We need to handle multiple events for the same program stage separately
                    event_groups = group.groupby("event")

                    # Track which version we're on for this program stage
                    version_counter = 1

                    # Process each event (repeated program stage) separately
                    for event_id, event_group in event_groups:
                        # Skip placeholder events
                        if str(event_id).startswith("placeholder_"):
                            continue

                        # Sort event group by data element to ensure consistent ordering
                        if not event_group.empty:
                            first_event_row = event_group.iloc[0]

                            # Create version suffix
                            version_suffix = (
                                f"_v{version_counter}" if version_counter > 1 else ""
                            )

                            # Add event metadata with version suffix
                            if "event" in first_event_row:
                                patient_row[
                                    f"event_{clean_stage_name}{version_suffix}"
                                ] = event_id

                            # Use eventDate as-is (original DHIS2 format)
                            if "eventDate" in first_event_row:
                                patient_row[
                                    f"event_date_{clean_stage_name}{version_suffix}"
                                ] = first_event_row.get("eventDate", "")

                            # Process all data elements for this specific event
                            for idx, row in event_group.iterrows():
                                data_element_name = row["dataElementName"]
                                value = row["value"]
                                data_element_uid = row.get("dataElement_uid", "")

                                # Skip empty values
                                if pd.isna(value) or str(value).strip() == "":
                                    continue

                                # Clean data element name
                                clean_element_name = CSVIntegration.clean_column_name(
                                    data_element_name
                                )

                                # SPECIAL FIX: For instrumental delivery, ensure proper naming
                                if (
                                    data_element_uid == "K8BCYRU1TUP"
                                ):  # Instrumental delivery element
                                    # Use a consistent name: instrumental_delivery_[stage_name]
                                    clean_element_name = "instrumental_delivery"

                                # Create column name with program stage AND version suffix
                                column_name = f"{clean_element_name}_{clean_stage_name}{version_suffix}"

                                # Add to patient row
                                patient_row[column_name] = value

                            # Increment version counter for next event in same program stage
                            version_counter += 1

                stage_patient_rows.append(patient_row)

            # Create DataFrame for this program stage
            if stage_patient_rows:
                stage_df = pd.DataFrame(stage_patient_rows)
                all_stage_data.append(stage_df)

        # Step 4: Merge all stage data with patient base
        if not all_stage_data:
            logger.warning("No program stage data found after transformation")
            return patient_base

        # Start with patient base and merge all program stage data
        patient_df = patient_base
        for stage_df in all_stage_data:
            patient_df = patient_df.merge(stage_df, on="tei_id", how="left")

        # Fill NaN values with empty string for better readability
        patient_df = patient_df.fillna("")

        # ========== POST-TRANSFORMATION FIXES ==========

        # 1. REORDER COLUMNS: Move "other number of newborns" next to "number of newborns"
        if program_type == "maternal":
            # Find all columns related to number of newborns
            newborn_count_cols = []
            other_newborn_cols = []

            for col in patient_df.columns:
                col_lower = col.lower()
                # Look for number of newborns columns (with or without version suffix)
                if (
                    "number_of_newborns" in col_lower
                    and "other_number" not in col_lower
                ):
                    newborn_count_cols.append(col)
                # Look for other number of newborns columns
                elif "other_number_of_newborns" in col_lower:
                    other_newborn_cols.append(col)

            # Reorder columns: for each number_of_newborns column, put corresponding other_number_of_newborns next
            if newborn_count_cols and other_newborn_cols:
                logger.info(
                    f"🔀 Reordering columns: {len(newborn_count_cols)} newborn count columns, {len(other_newborn_cols)} other newborn columns"
                )

                # Get current column order
                cols = list(patient_df.columns)

                for i, base_col in enumerate(newborn_count_cols):
                    # Find the matching other_newborn column (same version suffix)
                    # Extract version suffix from base_col
                    version_suffix = ""
                    if re.search(r"_v\d+$", base_col):
                        version_match = re.search(r"(_v\d+)$", base_col)
                        version_suffix = version_match.group(1) if version_match else ""

                    # Find matching other_newborn column
                    matching_other_col = None
                    for other_col in other_newborn_cols:
                        if other_col.endswith(version_suffix):
                            matching_other_col = other_col
                            break

                    if matching_other_col:
                        # Get positions of both columns
                        base_idx = cols.index(base_col)
                        other_idx = cols.index(matching_other_col)

                        # If other column is after base column, move it right after base
                        if other_idx > base_idx + 1:
                            # Remove other column from its current position
                            cols.pop(other_idx)
                            # Insert it right after base column
                            cols.insert(base_idx + 1, matching_other_col)
                            logger.info(
                                f"   ➡️  Moved '{matching_other_col}' after '{base_col}'"
                            )

                # Apply new column order
                patient_df = patient_df[cols]

        # 2. FIX INSTRUMENTAL DELIVERY: Convert "true"/"false" to "1"/"0" (only for maternal)
        if program_type == "maternal":
            instrumental_cols = [
                col
                for col in patient_df.columns
                if "instrumental_delivery" in col.lower()
            ]

            for col in instrumental_cols:
                if col in patient_df.columns:
                    # Convert boolean values
                    true_count = 0
                    false_count = 0
                    unchanged_count = 0

                    for idx, value in patient_df[col].items():
                        if isinstance(value, str):
                            value_lower = value.lower().strip()
                            if value_lower == "true":
                                patient_df.at[idx, col] = "1"
                                true_count += 1
                            elif value_lower == "false":
                                patient_df.at[idx, col] = "0"
                                false_count += 1
                            else:
                                unchanged_count += 1
                        elif pd.isna(value) or value == "":
                            unchanged_count += 1
                        else:
                            # Try to convert other types
                            try:
                                if str(value).lower().strip() == "true":
                                    patient_df.at[idx, col] = "1"
                                    true_count += 1
                                elif str(value).lower().strip() == "false":
                                    patient_df.at[idx, col] = "0"
                                    false_count += 1
                                else:
                                    unchanged_count += 1
                            except:
                                unchanged_count += 1

                    logger.info(
                        f"   🔢 {col}: Converted {true_count} 'true' to '1', {false_count} 'false' to '0', {unchanged_count} unchanged"
                    )

        # Add program type indicator
        patient_df["program_type"] = program_type

        # Log transformation statistics
        logger.info(f"✅ Transformation complete:")
        logger.info(
            f"   📊 Output: {len(patient_df)} patients, {len(patient_df.columns)} columns"
        )

        return patient_df

    @staticmethod
    def clean_transformed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the transformed dataframe AFTER transformation, BEFORE saving to CSV:
        1. Remove all has_actual_event columns
        2. Replace placeholder values with "N/A"
        3. Replace empty cells with "N/A"
        4. Replace empty event dates with enrollment date
        5. Fix instrumental delivery column names and values (maternal only)
        6. NEW: Replace "yes" in ARV Rx for Newborn columns with "N/A"
        """
        if df.empty:
            return df

        logger.info("🧹 CLEANING TRANSFORMED DATA - BEFORE SAVING TO CSV")
        cleaned_df = df.copy()

        # 1. REMOVE has_actual_event columns
        has_actual_cols = [
            col for col in cleaned_df.columns if col.startswith("has_actual_event_")
        ]
        if has_actual_cols:
            logger.info(f"🗑️ REMOVING {len(has_actual_cols)} has_actual_event columns")
            cleaned_df = cleaned_df.drop(columns=has_actual_cols)

        # 4. REPLACE EMPTY EVENT DATES WITH ENROLLMENT DATE
        # First, identify all event_date columns
        event_date_cols = [
            col for col in cleaned_df.columns if col.startswith("event_date_")
        ]

        if event_date_cols and "enrollment_date" in cleaned_df.columns:
            logger.info(f"📅 Filling empty event dates with enrollment date")
            event_date_filled_count = 0

            for event_date_col in event_date_cols:
                # Find rows where event_date is empty/N/A
                empty_mask = (
                    (cleaned_df[event_date_col].isna())
                    | (cleaned_df[event_date_col].astype(str).str.strip() == "")
                    | (cleaned_df[event_date_col].astype(str).str.strip() == "N/A")
                )

                # Find rows where enrollment_date has value
                enrollment_mask = (
                    cleaned_df["enrollment_date"].notna()
                    & (cleaned_df["enrollment_date"].astype(str).str.strip() != "")
                    & (cleaned_df["enrollment_date"].astype(str).str.strip() != "N/A")
                )

                # Apply enrollment_date where both conditions are met
                fill_mask = empty_mask & enrollment_mask
                fill_count = fill_mask.sum()

                if fill_count > 0:
                    cleaned_df.loc[fill_mask, event_date_col] = cleaned_df.loc[
                        fill_mask, "enrollment_date"
                    ]
                    event_date_filled_count += fill_count
                    logger.info(
                        f"   ✅ {event_date_col}: Filled {fill_count} empty dates with enrollment_date"
                    )

            logger.info(f"   📊 Total event dates filled: {event_date_filled_count}")

        # 2. REPLACE placeholder values with "N/A"
        placeholder_count = 0
        for col in cleaned_df.columns:
            # Check if column contains placeholder values
            if cleaned_df[col].dtype == "object":
                placeholder_mask = (
                    cleaned_df[col].astype(str).str.startswith("placeholder_")
                )
                if placeholder_mask.any():
                    placeholder_count += placeholder_mask.sum()
                    cleaned_df.loc[placeholder_mask, col] = "N/A"

        # 3. REPLACE empty cells with "N/A" (for non-date columns)
        empty_count = 0
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == "object":
                # Replace empty strings
                empty_mask = cleaned_df[col].astype(str).str.strip() == ""
                empty_count += empty_mask.sum()
                cleaned_df.loc[empty_mask, col] = "N/A"

                # Replace NaN
                nan_mask = cleaned_df[col].isna()
                empty_count += nan_mask.sum()
                cleaned_df.loc[nan_mask, col] = "N/A"

        # 6. NEW: REPLACE "yes" IN ARV Rx FOR NEWBORN COLUMNS WITH "N/A"
        arv_rx_cols = [
            col
            for col in cleaned_df.columns
            if "arv_rx" in col.lower() or "arv_rx_for_newborn" in col.lower()
        ]
        if arv_rx_cols:
            logger.info(f"💊 Processing ARV Rx for Newborn columns:")
            for col in arv_rx_cols:
                # Count "yes" values before replacement
                yes_mask = cleaned_df[col].astype(str).str.lower().str.strip() == "yes"
                yes_count = yes_mask.sum()

                if yes_count > 0:
                    # Replace "yes" with "N/A"
                    cleaned_df.loc[yes_mask, col] = "N/A"
                    logger.info(
                        f"   ✅ {col}: Replaced {yes_count} 'yes' values with 'N/A'"
                    )

                # Also check for case variations
                yes_variations = ["Yes", "YES", "yEs", "yeS"]
                for variation in yes_variations:
                    variation_mask = (
                        cleaned_df[col].astype(str).str.strip() == variation
                    )
                    variation_count = variation_mask.sum()
                    if variation_count > 0:
                        cleaned_df.loc[variation_mask, col] = "N/A"
                        logger.info(
                            f"   ✅ {col}: Also replaced {variation_count} '{variation}' values with 'N/A'"
                        )

        # ADDITIONAL CLEANUP: Ensure instrumental delivery columns are properly named (maternal only)
        instrumental_cols = [
            col
            for col in cleaned_df.columns
            if "instrumental" in col.lower() and "delivery" in col.lower()
        ]

        for col in instrumental_cols:
            # Ensure proper naming format
            correct_name = col
            # If column name has double "instrumental_delivery", fix it
            if "instrumental_delivery_instrumental_delivery" in col.lower():
                # Remove the duplicate part
                correct_name = re.sub(
                    r"instrumental_delivery_instrumental_delivery",
                    "instrumental_delivery",
                    col,
                    flags=re.IGNORECASE,
                )
                if correct_name != col:
                    # Rename the column
                    cleaned_df.rename(columns={col: correct_name}, inplace=True)
                    logger.info(f"   🔧 Renamed column: '{col}' -> '{correct_name}'")

        # Final check: ensure all boolean values in instrumental delivery are 1/0 (maternal only)
        for col in instrumental_cols:
            if col in cleaned_df.columns:
                # Convert any remaining "true"/"false" to "1"/"0"
                for idx, value in cleaned_df[col].items():
                    if isinstance(value, str):
                        value_lower = value.lower().strip()
                        if value_lower == "true":
                            cleaned_df.at[idx, col] = "1"
                        elif value_lower == "false":
                            cleaned_df.at[idx, col] = "0"

                # Count current values
                value_counts = cleaned_df[cleaned_df[col] != "N/A"][col].value_counts(
                    dropna=False
                )
                if not value_counts.empty:
                    logger.info(
                        f"   🔍 {col} final values: {dict(value_counts.head())}"
                    )

        # After filling with enrollment date, check if any event_date columns are still empty
        # and replace those with "N/A"
        if event_date_cols:
            event_date_empty_count = 0
            for event_date_col in event_date_cols:
                # Find rows where event_date is still empty
                still_empty_mask = (
                    (cleaned_df[event_date_col].isna())
                    | (cleaned_df[event_date_col].astype(str).str.strip() == "")
                    | (cleaned_df[event_date_col].astype(str).str.strip() == "N/A")
                )

                fill_count = still_empty_mask.sum()
                if fill_count > 0:
                    cleaned_df.loc[still_empty_mask, event_date_col] = "N/A"
                    event_date_empty_count += fill_count

            logger.info(
                f"   📅 Event dates still empty after enrollment fill: {event_date_empty_count}"
            )

        logger.info(f"✅ CLEANING DONE:")
        logger.info(f"   📤 Removed {len(has_actual_cols)} has_actual_event columns")
        logger.info(
            f"   🔄 Filled event dates with enrollment_date: {event_date_filled_count}"
        )
        logger.info(f"   🔄 Replaced {placeholder_count} placeholder values with 'N/A'")
        logger.info(f"   🔄 Replaced {empty_count} empty/NaN cells with 'N/A'")
        logger.info(
            f"   💊 Processed ARV Rx for Newborn columns: {len(arv_rx_cols)} columns"
        )

        return cleaned_df

    @staticmethod
    def post_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process the DataFrame AFTER transformation, BEFORE saving to CSV
        Simple version: just apply cleaning and keep dates as-is
        """
        if df.empty:
            return df

        logger.info("📦 POST-PROCESSING TRANSFORMED DATA")

        # Step 1: Apply cleaning (remove has_actual_event, replace placeholders/empty, fill event dates)
        processed_df = CSVIntegration.clean_transformed_dataframe(df)

        # Check if this is maternal or newborn program
        is_maternal = (
            "program_type" in processed_df.columns
            and processed_df["program_type"].iloc[0] == "maternal"
        )

        if is_maternal:
            # VERIFICATION: Check column ordering for newborn columns
            newborn_cols = [
                col for col in processed_df.columns if "newborns" in col.lower()
            ]
            if newborn_cols:
                logger.info("👶 Newborn column verification:")
                for i, col in enumerate(newborn_cols):
                    logger.info(f"   {i+1}. {col}")

            # VERIFICATION: Check instrumental delivery values
            instrumental_cols = [
                col
                for col in processed_df.columns
                if "instrumental_delivery" in col.lower()
            ]
            if instrumental_cols:
                logger.info("✂️ Instrumental delivery verification:")
                for col in instrumental_cols:
                    value_counts = processed_df[processed_df[col] != "N/A"][
                        col
                    ].value_counts()
                    total = processed_df[processed_df[col] != "N/A"].shape[0]
                    if not value_counts.empty:
                        logger.info(f"   {col}: {total} values -> {dict(value_counts)}")

            # VERIFICATION: Check ARV Rx for Newborn columns
            arv_rx_cols = [
                col
                for col in processed_df.columns
                if "arv_rx" in col.lower() or "arv_rx_for_newborn" in col.lower()
            ]
            if arv_rx_cols:
                logger.info("💊 ARV Rx for Newborn column verification:")
                for col in arv_rx_cols:
                    value_counts = processed_df[processed_df[col] != "N/A"][
                        col
                    ].value_counts()
                    total = processed_df[processed_df[col] != "N/A"].shape[0]
                    if not value_counts.empty:
                        logger.info(f"   {col}: {total} values -> {dict(value_counts)}")
                    # Also check that there are no "yes" values remaining
                    yes_count = processed_df[
                        processed_df[col].astype(str).str.lower().str.strip() == "yes"
                    ].shape[0]
                    if yes_count > 0:
                        logger.warning(
                            f"   ⚠️ Found {yes_count} 'yes' values in {col} - should have been replaced with N/A"
                        )
        else:
            # NEWBORN PROGRAM VERIFICATIONS
            logger.info("👶 Newborn program verification:")

            # Check for birth weight column
            birth_weight_cols = [
                col for col in processed_df.columns if "birth_weight" in col.lower()
            ]
            if birth_weight_cols:
                logger.info(f"   ⚖️ Birth weight columns found: {birth_weight_cols}")
                for col in birth_weight_cols:
                    non_empty = processed_df[processed_df[col] != "N/A"].shape[0]
                    total = processed_df.shape[0]
                    logger.info(
                        f"      - {col}: {non_empty}/{total} non-empty values ({non_empty/total*100:.1f}%)"
                    )
            else:
                logger.warning("   ⚠️ No birth weight columns found - check mapping")

            # Check for any versioned columns (should not have any)
            versioned_cols = [
                col for col in processed_df.columns if re.search(r"_v\d+$", col)
            ]
            if versioned_cols:
                logger.warning(
                    f"   ⚠️ Found {len(versioned_cols)} versioned columns in newborn program (should not have any)"
                )
                for col in versioned_cols[:5]:
                    logger.warning(f"      - {col}")

        logger.info(f"✅ POST-PROCESSING COMPLETE")
        logger.info(
            f"   📊 Final shape: {len(processed_df)} rows, {len(processed_df.columns)} columns"
        )

        # Show event date filling statistics
        event_date_cols = [
            col for col in processed_df.columns if col.startswith("event_date_")
        ]
        if event_date_cols:
            logger.info("📅 Event date statistics:")
            for col in event_date_cols[:3]:  # Show first 3
                filled = processed_df[processed_df[col] != "N/A"].shape[0]
                total = processed_df.shape[0]

                logger.info(
                    f"   {col}: {filled}/{total} filled ({filled/total*100:.1f}%)"
                )

        return processed_df


# ========== AUTOMATED PIPELINE CLASS ==========


class AutomatedDHIS2Pipeline:
    """Automated pipeline to process both maternal and newborn programs"""

    def __init__(
        self,
        base_url: str = None,
        username: str = None,
        password: str = None,
        csv_path: str = None,
        output_base_dir: str = None,
    ):
        """
        Initialize automated pipeline

        Args:
            base_url: DHIS2 base URL (optional - tries config.py first)
            username: DHIS2 username (optional - tries config.py first)
            password: DHIS2 password (optional - tries config.py first)
            csv_path: Path to maternal CSV file (optional)
            output_base_dir: Base directory for output (optional)
        """
        # Try to get from config.py if not provided
        if not all([base_url, username, password]):
            try:
                import sys

                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(script_dir))
                sys.path.insert(0, project_root)

                from config import settings

                base_url = base_url or settings.DHIS2_BASE_URL
                username = username or settings.DHIS2_USERNAME
                password = password or settings.DHIS2_PASSWORD
                logger.info("✅ Using credentials from config.py")
            except Exception as e:
                logger.warning(f"⚠️ Could not load from config.py: {e}")

        self.base_url = base_url
        self.username = username
        self.password = password
        self.csv_path = csv_path

        # FIX 1: Set output directories correctly
        if output_base_dir:
            self.output_base_dir = output_base_dir
        else:
            # Use the corrected DEFAULT_OUTPUT_DIR
            self.output_base_dir = DEFAULT_OUTPUT_DIR

        # Ensure the output directory exists
        os.makedirs(self.output_base_dir, exist_ok=True)

        self.maternal_dir = os.path.join(self.output_base_dir, "maternal")
        self.newborn_dir = os.path.join(self.output_base_dir, "newborn")

        # Create directories
        os.makedirs(self.maternal_dir, exist_ok=True)
        os.makedirs(self.newborn_dir, exist_ok=True)

        # Initialize fetcher
        self.fetcher = DHIS2DataFetcher(base_url, username, password)
        self.facility_to_region_map = {}
        self.csv_data = None

        logger.info("=" * 80)
        logger.info("AUTOMATED DHIS2 PIPELINE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"DHIS2 URL: {self.base_url}")
        logger.info(f"Username: {self.username}")
        logger.info(f"Output Directory: {self.output_base_dir}")
        logger.info(f"Maternal Directory: {self.maternal_dir}")
        logger.info(f"Newborn Directory: {self.newborn_dir}")
        logger.info("=" * 80)

        # Log absolute paths for debugging
        logger.info(f"📁 Absolute paths:")
        logger.info(f"   Script dir: {SCRIPT_DIR}")
        logger.info(f"   Dashboard utils dir: {DASHBOARD_UTILS_DIR}")
        logger.info(f"   Default output dir: {DEFAULT_OUTPUT_DIR}")
        logger.info(f"   Actual output dir: {os.path.abspath(self.output_base_dir)}")

    def process_program(self, program_uid: str, program_name: str) -> bool:
        """
        Process a single program (maternal or newborn)
        FIXED: Added better logging and verification
        """
        logger.info("=" * 80)
        logger.info(f"🔄 PROCESSING {program_name.upper()} PROGRAM")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch orgUnit names
            logger.info("📋 Fetching orgUnit names...")
            orgunit_names = self.fetcher.fetch_orgunit_names()
            logger.info(f"✅ Fetched {len(orgunit_names)} orgUnit names")

            # Step 2: Fetch all regions
            logger.info("🗺️ Fetching regions...")
            regions = self.fetcher.fetch_all_regions()
            if not regions:
                logger.error("❌ No regions found")
                return False

            total_regions = len(regions)
            logger.info(f"📊 Found {total_regions} regions")

            # Step 3: Process each region
            all_patient_data = []
            regional_files_created = 0
            total_teis_fetched = 0

            for i, (region_uid, region_name) in enumerate(regions.items(), 1):
                logger.info(f"\n🏞️ [{i}/{total_regions}] Processing: {region_name}")

                # Fetch program data with better logging
                logger.info(f"   📥 Fetching TEI data for region...")
                tei_data = self.fetcher.fetch_program_data(
                    program_uid, region_uid, "DESCENDANTS", 1000
                )

                tei_count = len(tei_data.get("trackedEntityInstances", []))
                total_teis_fetched += tei_count
                logger.info(f"   📊 Found {tei_count} TEIs in this region")

                if tei_count == 0:
                    continue

                # Create events dataframe
                events_df = CSVIntegration.create_events_dataframe(
                    tei_data, program_uid, orgunit_names
                )
                logger.info(
                    f"   📈 Created {len(events_df)} events from {tei_count} TEIs"
                )

                # For maternal program only: integrate CSV data
                if program_uid == MATERNAL_PROGRAM_UID and self.csv_data is not None:
                    logger.info("   📁 Integrating CSV data...")
                    region_csv = CSVIntegration.filter_csv_data_by_user_access(
                        self.csv_data,
                        "regional",
                        region_name,
                        self.facility_to_region_map,
                    )

                    if not region_csv.empty:
                        logger.info(
                            f"   📊 Found {len(region_csv)} CSV rows for this region"
                        )
                        events_df = (
                            CSVIntegration.integrate_maternal_csv_data_for_region(
                                events_df, region_csv, region_name
                            )
                        )
                        logger.info(
                            f"   🔄 After CSV integration: {len(events_df)} events"
                        )

                # Transform to patient-level
                patient_df = CSVIntegration.transform_events_to_patient_level(
                    events_df, program_uid
                )

                if not patient_df.empty:
                    # Post-process
                    patient_df = CSVIntegration.post_process_dataframe(patient_df)
                    patient_df["region_uid"] = region_uid
                    patient_df["region_name"] = region_name
                    all_patient_data.append(patient_df)

                    logger.info(f"   ✅ Processed {len(patient_df)} patients")

                    # Save individual regional file
                    saved_path = self.save_region_file(
                        patient_df, region_name, program_uid
                    )
                    if saved_path:
                        regional_files_created += 1
                        logger.info(f"   💾 Saved to: {os.path.basename(saved_path)}")

            # Log total TEIs fetched
            logger.info(
                f"\n📊 TOTAL TEIs FETCHED FOR {program_name.upper()}: {total_teis_fetched}"
            )

            # Step 4: Combine all regions
            if not all_patient_data:
                logger.warning(f"⚠️ No data processed for {program_name}")
                return False

            combined_df = pd.concat(all_patient_data, ignore_index=True)

            # Step 5: Save national file to appropriate directory
            program_type = (
                "maternal" if program_uid == MATERNAL_PROGRAM_UID else "newborn"
            )
            if program_type == "maternal":
                output_dir = self.maternal_dir
            else:
                output_dir = self.newborn_dir

            # Save national file
            national_file = os.path.join(output_dir, f"national_{program_type}.csv")
            combined_df.to_csv(national_file, index=False, encoding="utf-8")

            logger.info("=" * 80)
            logger.info(f"✅ {program_name.upper()} PROCESSING COMPLETE")
            logger.info(f"📊 Total patients in combined file: {len(combined_df)}")
            logger.info(f"🏞️ Total regions: {combined_df['region_name'].nunique()}")
            logger.info(f"📁 Regional files created: {regional_files_created}")
            logger.info(f"🌍 National file: {national_file}")
            logger.info(f"📁 Absolute path: {os.path.abspath(national_file)}")

            # Verify file was created
            if os.path.exists(national_file):
                file_size = os.path.getsize(national_file)
                logger.info(f"📦 File size: {file_size:,} bytes")
            else:
                logger.error(f"❌ National file was not created: {national_file}")

            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ Error processing {program_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_csv_data(self):
        """Load and map CSV data for maternal program"""
        if not self.csv_path or not os.path.exists(self.csv_path):
            logger.warning(f"⚠️ CSV file not found: {self.csv_path}")
            return False

        try:
            logger.info(f"📂 Loading CSV data: {self.csv_path}")
            self.csv_data = pd.read_csv(self.csv_path)
            logger.info(
                f"✅ Loaded CSV: {len(self.csv_data)} rows, {len(self.csv_data['tei_id'].unique())} unique TEIs"
            )

            # Fetch facility-to-region mapping
            logger.info("🗺️ Fetching facility-to-region mapping...")
            self.facility_to_region_map = (
                self.fetcher.fetch_facility_to_region_mapping()
            )

            if self.facility_to_region_map:
                logger.info(
                    f"✅ Fetched mapping for {len(self.facility_to_region_map)} facilities"
                )
                self.csv_data = CSVIntegration.get_facility_region_mapping(
                    self.csv_data, self.facility_to_region_map
                )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {str(e)}")
            self.csv_data = None
            return False

    def save_region_file(
        self, patient_df: pd.DataFrame, region_name: str, program_uid: str
    ):
        """Save individual region file with new naming format: regional_{Region name}_programtype.csv"""
        try:
            # Clean region name for filename
            clean_region = re.sub(r"[^\w\s-]", "", region_name)
            clean_region = re.sub(r"[-\s]+", "_", clean_region)

            # Generate filename
            program_type = (
                "maternal" if program_uid == MATERNAL_PROGRAM_UID else "newborn"
            )
            filename = f"regional_{clean_region}_{program_type}.csv"

            # Save to appropriate directory
            if program_type == "maternal":
                output_dir = self.maternal_dir
            else:
                output_dir = self.newborn_dir

            filepath = os.path.join(output_dir, filename)

            # Save file (without region columns for cleaner regional files)
            save_df = patient_df.drop(
                columns=["region_uid", "region_name"], errors="ignore"
            )
            save_df.to_csv(filepath, index=False, encoding="utf-8")

            logger.info(
                f"   💾 Saved regional file: {filename} ({len(patient_df)} patients)"
            )

            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving region file: {str(e)}")
            return None

    def run_pipeline(self) -> bool:
        """
        Run complete pipeline for both programs

        Returns:
            True if both programs successful, False otherwise
        """
        logger.info("🚀 STARTING COMPLETE AUTOMATED PIPELINE")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("=" * 80)

        # Step 1: Test connection
        logger.info("🔗 Testing DHIS2 connection...")
        regions = self.fetcher.fetch_all_regions()
        if not regions:
            logger.error("❌ Failed to connect to DHIS2")
            return False

        logger.info(f"✅ Connected to DHIS2 successfully")
        logger.info(f"   Found {len(regions)} regions")

        # Step 2: Load CSV data (for maternal only)
        if self.csv_path:
            self.load_csv_data()

        # Step 3: Process MATERNAL program
        maternal_success = self.process_program(MATERNAL_PROGRAM_UID, "MATERNAL")

        # Step 4: Process NEWBORN program
        newborn_success = self.process_program(NEWBORN_PROGRAM_UID, "NEWBORN")

        # Summary
        logger.info("=" * 80)
        logger.info("📊 PIPELINE COMPLETION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Maternal: {'✅ SUCCESS' if maternal_success else '❌ FAILED'}")
        logger.info(f"Newborn: {'✅ SUCCESS' if newborn_success else '❌ FAILED'}")
        logger.info(f"End time: {datetime.now()}")
        logger.info("=" * 80)

        return maternal_success and newborn_success


# ========== GUI APPLICATION ==========


class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack everything
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling
        self.bind_mouse_wheel()

    def bind_mouse_wheel(self):
        """Bind mouse wheel for scrolling"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")


class DHIS2DataFetcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DHIS2 Data Fetcher with CSV Integration - UPDATED")
        self.root.geometry("1200x800")

        # Set minimum size
        self.root.minsize(1000, 700)

        # Variables
        self.base_url = tk.StringVar(value="https://imnid.aau.edu.et/dhis")
        self.username = tk.StringVar(value="")
        self.password = tk.StringVar(value="")
        self.level_choice = tk.StringVar(value="national")
        self.program_choice = tk.StringVar(value="maternal")
        self.program_uid = tk.StringVar(value=MATERNAL_PROGRAM_UID)  # Updated
        self.region_uid = tk.StringVar()
        self.region_name = tk.StringVar()
        self.csv_path = tk.StringVar(value="maternal_data_long_format.csv")
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.page_size = tk.IntVar(value=1000)
        self.status_queue = Queue()
        self.is_fetching = False
        self.national_data = []  # Store all national data for combined CSV
        self.facility_to_region_map = {}  # Store facility to region mapping
        self.csv_data = None  # Store loaded CSV data with region mapping

        # Initialize fetcher
        self.fetcher = None

        # Create UI
        self.create_widgets()

        # Start status updater
        self.root.after(100, self.update_status)

        # Center window
        self.center_window()

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        # Create main container with scrollbar
        main_container = ScrollableFrame(self.root)
        main_container.pack(fill="both", expand=True)

        # Use the scrollable frame's inner frame
        main_frame = main_container.scrollable_frame

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="DHIS2 Data Fetcher with CSV Integration - UPDATED",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="File naming: regional_{Region name}_programtype.csv and national_programtype.csv | Dates in original DHIS2 format | ARV Rx 'yes' → 'N/A' | Newborn: No version suffixes",
            font=("Arial", 10, "italic"),
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)

        row = 2

        # DHIS2 Configuration
        config_frame = ttk.LabelFrame(
            main_frame, text="DHIS2 Configuration", padding="15"
        )
        config_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Base URL:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        ttk.Entry(config_frame, textvariable=self.base_url, width=70).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5, columnspan=2
        )

        ttk.Label(config_frame, text="Username:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        ttk.Entry(config_frame, textvariable=self.username).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Button(
            config_frame, text="Test Connection", command=self.test_connection, width=15
        ).grid(row=1, column=2, padx=(10, 0), pady=5)

        ttk.Label(config_frame, text="Password:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        ttk.Entry(config_frame, textvariable=self.password, show="*").grid(
            row=2, column=1, sticky=(tk.W, tk.E), pady=5
        )

        ttk.Label(config_frame, text="Page Size:").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 5)
        )
        page_size_frame = ttk.Frame(config_frame)
        page_size_frame.grid(row=3, column=1, sticky=tk.W, pady=(10, 5))
        ttk.Entry(page_size_frame, textvariable=self.page_size, width=10).pack(
            side=tk.LEFT
        )
        ttk.Label(page_size_frame, text=" (TEIs per request)").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        row += 1

        # Program Selection
        program_frame = ttk.LabelFrame(
            main_frame, text="Program Selection", padding="15"
        )
        program_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Program type radio buttons
        program_type_frame = ttk.Frame(program_frame)
        program_type_frame.grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10)
        )

        ttk.Label(program_type_frame, text="Select Program Type:").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Radiobutton(
            program_type_frame,
            text="Maternal Health Program",
            variable=self.program_choice,
            value="maternal",
            command=self.on_program_change,
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            program_type_frame,
            text="Newborn Care Program",
            variable=self.program_choice,
            value="newborn",
            command=self.on_program_change,
        ).pack(side=tk.LEFT)

        # Program UID (automatically set based on selection)
        ttk.Label(program_frame, text="Program UID:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        ttk.Entry(program_frame, textvariable=self.program_uid, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5, columnspan=2
        )

        # CSV Path (for maternal only)
        self.csv_frame = ttk.Frame(program_frame)
        self.csv_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(self.csv_frame, text="CSV File Path:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        ttk.Entry(self.csv_frame, textvariable=self.csv_path, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E)
        )
        ttk.Button(
            self.csv_frame, text="Browse...", command=self.browse_csv_file, width=10
        ).grid(row=0, column=2, padx=(10, 0))

        ttk.Button(
            program_frame,
            text="📊 Load & Map CSV Facilities",
            command=self.load_and_map_csv,
            width=25,
        ).grid(row=3, column=0, columnspan=3, pady=(10, 5))

        # Example UIDs
        example_frame = ttk.Frame(program_frame)
        example_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        ttk.Label(
            example_frame, text="Example UIDs:", font=("Arial", 9, "italic")
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            example_frame,
            text=f"Maternal: {MATERNAL_PROGRAM_UID}",
            font=("Arial", 9, "italic"),
            foreground="blue",
        ).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(
            example_frame,
            text=f"Newborn: {NEWBORN_PROGRAM_UID}",
            font=("Arial", 9, "italic"),
            foreground="green",
        ).pack(side=tk.LEFT)

        # NEW: Automated Pipeline Button
        ttk.Button(
            program_frame,
            text="🤖 Run Automated Pipeline (Both Programs)",
            command=self.run_automated_pipeline,
            width=35,
        ).grid(row=5, column=0, columnspan=3, pady=(15, 5))

        row += 1

        # Level Selection
        level_frame = ttk.LabelFrame(
            main_frame, text="Data Level Selection", padding="15"
        )
        level_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Level radio buttons
        level_type_frame = ttk.Frame(level_frame)
        level_type_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        ttk.Label(level_type_frame, text="Select Data Level:").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Radiobutton(
            level_type_frame,
            text="National Level (All regions)",
            variable=self.level_choice,
            value="national",
            command=self.on_level_change,
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            level_type_frame,
            text="Regional Level (Single region)",
            variable=self.level_choice,
            value="regional",
            command=self.on_level_change,
        ).pack(side=tk.LEFT)

        # Region Selection (hidden by default)
        self.region_frame = ttk.Frame(level_frame)
        self.region_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(self.region_frame, text="Region Name:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        ttk.Entry(self.region_frame, textvariable=self.region_name, width=40).grid(
            row=0, column=1, sticky=(tk.W, tk.E)
        )

        ttk.Label(self.region_frame, text="Region UID:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0)
        )
        ttk.Entry(self.region_frame, textvariable=self.region_uid, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0)
        )

        ttk.Button(
            self.region_frame,
            text="Fetch Regions",
            command=self.fetch_regions,
            width=15,
        ).grid(row=0, column=2, rowspan=2, padx=(10, 0))

        # Hide region frame initially
        self.region_frame.grid_remove()

        row += 1

        # Output Directory
        output_frame = ttk.LabelFrame(
            main_frame, text="Output Configuration", padding="15"
        )
        output_frame.grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        ttk.Entry(output_frame, textvariable=self.output_dir, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E)
        )
        ttk.Button(
            output_frame, text="Browse...", command=self.browse_output_dir, width=10
        ).grid(row=0, column=2, padx=(10, 0))

        # NEW: Create Program Folders Button
        ttk.Button(
            output_frame,
            text="📁 Create Maternal/Newborn Folders",
            command=self.create_program_folders,
            width=30,
        ).grid(row=1, column=0, columnspan=3, pady=(10, 0))

        row += 1

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 20))

        self.fetch_button = ttk.Button(
            button_frame,
            text="🚀 Fetch & Transform Data",
            command=self.start_fetching,
            width=25,
        )
        self.fetch_button.grid(row=0, column=0, padx=(0, 10))

        ttk.Button(
            button_frame, text="🗑️ Clear Log", command=self.clear_log, width=15
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(
            button_frame,
            text="📂 Open Output Folder",
            command=self.open_output_folder,
            width=20,
        ).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Exit", command=self.root.quit, width=15).grid(
            row=0, column=3
        )

        row += 1

        # Status/Progress
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="15")
        status_frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky=(tk.W, tk.E, tk.N, tk.S),
            pady=(0, 20),
        )
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        # Status text with scrollbar
        self.status_text = scrolledtext.ScrolledText(
            status_frame, height=15, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, mode="indeterminate", length=400
        )
        self.progress_bar.grid(
            row=row + 1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 10)
        )

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Arial", 9))
        self.status_label.grid(row=row + 2, column=0, columnspan=3, sticky=tk.W)

    def on_program_change(self):
        """Show/hide CSV path based on program choice"""
        if self.program_choice.get() == "maternal":
            self.csv_frame.grid()
            self.program_uid.set(MATERNAL_PROGRAM_UID)
        else:
            self.csv_frame.grid_remove()
            self.program_uid.set(NEWBORN_PROGRAM_UID)

    def on_level_change(self):
        """Show/hide region selection based on level choice"""
        if self.level_choice.get() == "regional":
            self.region_frame.grid()
        else:
            self.region_frame.grid_remove()

    def create_program_folders(self):
        """Create maternal and newborn folders"""
        output_dir = self.output_dir.get()
        maternal_dir = os.path.join(output_dir, "maternal")
        newborn_dir = os.path.join(output_dir, "newborn")

        try:
            os.makedirs(maternal_dir, exist_ok=True)
            os.makedirs(newborn_dir, exist_ok=True)

            self.log_message(f"✅ Created folders:")
            self.log_message(f"   📁 {maternal_dir}")
            self.log_message(f"   📁 {newborn_dir}")
            messagebox.showinfo(
                "Folders Created",
                f"Successfully created program folders:\n\n"
                f"Maternal: {maternal_dir}\n"
                f"Newborn: {newborn_dir}",
            )
        except Exception as e:
            self.log_message(f"❌ Failed to create folders: {str(e)}")
            messagebox.showerror("Error", f"Failed to create folders:\n\n{str(e)}")

    def run_automated_pipeline(self):
        """Run automated pipeline for both programs"""
        if not self.validate_credentials():
            return

        if self.is_fetching:
            messagebox.showwarning(
                "Already Running", "Data fetching is already in progress."
            )
            return

        # Disable fetch button
        self.is_fetching = True
        self.fetch_button.config(
            state=tk.DISABLED, text="⏳ Running Automated Pipeline..."
        )
        self.progress_bar.start()

        # Start automated pipeline in background thread
        thread = threading.Thread(
            target=self.run_automated_pipeline_thread, daemon=True
        )
        thread.start()

    def run_automated_pipeline_thread(self):
        """Thread function for automated pipeline"""
        try:
            self.log_message("=" * 80)
            self.log_message("🤖 STARTING AUTOMATED PIPELINE")
            self.log_message("=" * 80)
            self.log_message("This will process BOTH maternal and newborn programs")
            self.log_message(f"Output directory: {self.output_dir.get()}")
            self.log_message("=" * 80)

            # Create pipeline
            pipeline = AutomatedDHIS2Pipeline(
                base_url=self.base_url.get(),
                username=self.username.get(),
                password=self.password.get(),
                csv_path=(
                    self.csv_path.get()
                    if self.program_choice.get() == "maternal"
                    else None
                ),
                output_base_dir=self.output_dir.get(),
            )

            # Run pipeline
            success = pipeline.run_pipeline()

            if success:
                self.status_queue.put(
                    (
                        "success",
                        f"✅ AUTOMATED PIPELINE COMPLETE!\n\n"
                        f"Files generated for ALL regions:\n"
                        f"• {pipeline.maternal_dir}\\regional_*_maternal.csv (one per region)\n"
                        f"• {pipeline.newborn_dir}\\regional_*_newborn.csv (one per region)\n\n"
                        f"Plus national files:\n"
                        f"• {pipeline.maternal_dir}\\national_maternal.csv\n"
                        f"• {pipeline.newborn_dir}\\national_newborn.csv\n\n"
                        f"Total files: All regional files + 2 national files",
                    )
                )
            else:
                self.status_queue.put(
                    ("error", "Automated pipeline failed. Check logs for details.")
                )

        except Exception as e:
            self.log_message(f"❌ Error in automated pipeline: {str(e)}")
            self.log_message(traceback.format_exc())
            self.status_queue.put(("error", f"Automated pipeline error: {str(e)}"))
        finally:
            self.status_queue.put(("finished", None))

    def browse_csv_file(self):
        """Browse for CSV file"""
        initial_dir = (
            os.path.dirname(self.csv_path.get())
            if os.path.exists(self.csv_path.get())
            else os.getcwd()
        )
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=initial_dir,
        )
        if filename:
            self.csv_path.set(filename)
            self.log_message(f"Selected CSV file: {filename}")

    def browse_output_dir(self):
        """Browse for output directory"""
        initial_dir = (
            self.output_dir.get()
            if os.path.exists(self.output_dir.get())
            else os.getcwd()
        )
        directory = filedialog.askdirectory(
            title="Select Output Directory", initialdir=initial_dir
        )
        if directory:
            self.output_dir.set(directory)
            self.log_message(f"Output directory set to: {directory}")

    def load_and_map_csv(self):
        """Load CSV file and map facilities to regions"""
        if not self.validate_credentials():
            return

        csv_path = self.csv_path.get()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showwarning(
                "CSV Not Found",
                f"CSV file not found at:\n{csv_path}\n\nPlease select a valid CSV file.",
            )
            return

        try:
            self.log_message(f"📂 Loading CSV file: {csv_path}")
            self.update_status_label("Loading CSV data...")

            # Load CSV data
            self.csv_data = pd.read_csv(csv_path)
            self.log_message(
                f"✅ Loaded CSV data: {len(self.csv_data)} rows, {len(self.csv_data['tei_id'].unique())} unique TEIs"
            )

            # Initialize fetcher
            self.fetcher = DHIS2DataFetcher(
                self.base_url.get(), self.username.get(), self.password.get()
            )

            # Fetch facility-to-region mapping
            self.log_message("🗺️ Fetching facility-to-region mapping from DHIS2...")
            self.update_status_label("Fetching facility mapping...")

            self.facility_to_region_map = (
                self.fetcher.fetch_facility_to_region_mapping()
            )

            if not self.facility_to_region_map:
                messagebox.showwarning(
                    "No Facility Mapping",
                    "Could not fetch facility-to-region mapping from DHIS2.\n\n"
                    "CSV integration will use direct matching by orgUnit_name.",
                )
                self.log_message("⚠️ Could not fetch facility-to-region mapping")
            else:
                self.log_message(
                    f"✅ Fetched mapping for {len(self.facility_to_region_map)} facilities"
                )

                # Map CSV facilities to regions
                self.csv_data = CSVIntegration.get_facility_region_mapping(
                    self.csv_data, self.facility_to_region_map
                )

                # Show mapping statistics
                if "csv_region" in self.csv_data.columns:
                    region_counts = self.csv_data["csv_region"].value_counts()
                    self.log_message("📊 CSV Region Distribution:")
                    for region, count in region_counts.items():
                        if region and region != "":
                            self.log_message(f"   {region}: {count} rows")

                    unmapped = self.csv_data[self.csv_data["csv_region"] == ""].shape[0]
                    self.log_message(f"   Unmapped: {unmapped} rows")

            self.update_status_label("CSV loaded and mapped")
            messagebox.showinfo(
                "CSV Loaded",
                f"Successfully loaded and mapped CSV data!\n\n"
                f"Rows: {len(self.csv_data)}\n"
                f"Unique TEIs: {len(self.csv_data['tei_id'].unique())}\n"
                f"Mapped facilities: {len(self.facility_to_region_map)}",
            )

        except Exception as e:
            messagebox.showerror(
                "Error Loading CSV", f"Failed to load CSV file:\n\n{str(e)}"
            )
            self.log_message(f"❌ Error loading CSV: {str(e)}")
            self.update_status_label("CSV loading failed")

    def test_connection(self):
        """Test DHIS2 connection"""
        if not self.validate_credentials():
            return

        try:
            self.fetcher = DHIS2DataFetcher(
                self.base_url.get(), self.username.get(), self.password.get()
            )

            # Try to fetch regions to test connection
            regions = self.fetcher.fetch_all_regions()

            if regions:
                messagebox.showinfo(
                    "Connection Successful",
                    f"✅ Successfully connected to DHIS2!\n\n"
                    f"Base URL: {self.base_url.get()}\n"
                    f"Found {len(regions)} regions.\n\n"
                    f"Sample regions:\n" + "\n".join(list(regions.values())[:3]),
                )
                self.log_message(
                    f"✅ Connection successful - Found {len(regions)} regions"
                )
                self.update_status_label("Connection successful")
            else:
                messagebox.showwarning(
                    "Connection Warning",
                    "Connected but no regions found or API returned empty.",
                )
                self.log_message("⚠️ Connected but no regions found")

        except Exception as e:
            messagebox.showerror(
                "Connection Failed",
                f"❌ Failed to connect to DHIS2:\n\n{str(e)}\n\n"
                f"Please check:\n"
                f"1. Base URL is correct\n"
                f"2. Username and password are correct\n"
                f"3. You have internet connection",
            )
            self.log_message(f"❌ Connection failed: {str(e)}")
            self.update_status_label("Connection failed")

    def fetch_regions(self):
        """Fetch regions from DHIS2 and show in dialog"""
        if not self.validate_credentials():
            return

        try:
            self.fetcher = DHIS2DataFetcher(
                self.base_url.get(), self.username.get(), self.password.get()
            )

            self.log_message("Fetching regions from DHIS2...")
            self.update_status_label("Fetching regions...")

            regions = self.fetcher.fetch_all_regions()

            if regions:
                self.show_regions_dialog(regions)
                self.log_message(f"✅ Fetched {len(regions)} regions")
                self.update_status_label(f"Fetched {len(regions)} regions")
            else:
                messagebox.showwarning(
                    "No Regions", "No regions found or failed to fetch regions."
                )
                self.log_message("⚠️ No regions found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch regions: {str(e)}")
            self.log_message(f"❌ Failed to fetch regions: {str(e)}")

    def show_regions_dialog(self, regions: Dict[str, str]):
        """Show dialog with list of regions"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Region")
        dialog.geometry("700x600")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Search frame
        search_frame = ttk.Frame(dialog, padding="10")
        search_frame.pack(fill=tk.X)

        ttk.Label(search_frame, text="🔍 Search:").pack(side=tk.LEFT, padx=(0, 5))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10)
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=listbox.yview)

        # Sort regions by name
        sorted_regions = sorted(regions.items(), key=lambda x: x[1])

        for uid, name in sorted_regions:
            listbox.insert(tk.END, f"{name} ({uid})")

        # Selection buttons
        button_frame = ttk.Frame(dialog, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(
            button_frame,
            text="Select",
            command=lambda: self.select_region_from_dialog(dialog, listbox, regions),
            width=15,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(
            side=tk.LEFT
        )

        # Bind search
        def on_search_changed(*args):
            search_term = search_var.get().lower()
            listbox.delete(0, tk.END)
            for uid, name in sorted_regions:
                if search_term in name.lower() or search_term in uid.lower():
                    listbox.insert(tk.END, f"{name} ({uid})")

        search_var.trace("w", on_search_changed)

        # Bind double-click
        listbox.bind(
            "<Double-Button-1>",
            lambda e: self.select_region_from_dialog(dialog, listbox, regions),
        )

    def select_region_from_dialog(self, dialog, listbox, regions):
        """Handle region selection from dialog"""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a region.")
            return

        selected_text = listbox.get(selection[0])
        # Extract region name and UID from the text
        # Format: "Region Name (uid)"
        match = re.match(r"(.+) \(([^)]+)\)", selected_text)
        if match:
            region_name = match.group(1)
            region_uid = match.group(2)

            self.region_name.set(region_name)
            self.region_uid.set(region_uid)

            self.log_message(f"Selected region: {region_name} ({region_uid})")
            dialog.destroy()

    def validate_credentials(self):
        """Validate input credentials"""
        if not self.base_url.get():
            messagebox.showwarning("Missing URL", "Please enter DHIS2 base URL.")
            return False
        if not self.username.get():
            messagebox.showwarning("Missing Username", "Please enter DHIS2 username.")
            return False
        if not self.password.get():
            messagebox.showwarning("Missing Password", "Please enter DHIS2 password.")
            return False
        return True

    def start_fetching(self):
        """Start data fetching in a separate thread"""
        if not self.validate_credentials():
            return

        if self.is_fetching:
            messagebox.showwarning(
                "Already Running", "Data fetching is already in progress."
            )
            return

        # Disable fetch button
        self.is_fetching = True
        self.fetch_button.config(state=tk.DISABLED, text="⏳ Fetching...")
        self.progress_bar.start()

        # Start fetching in background thread
        thread = threading.Thread(target=self.fetch_data_thread, daemon=True)
        thread.start()

    def fetch_data_thread(self):
        """Thread function for fetching data"""
        try:
            self.log_message("=" * 80)
            self.log_message("🚀 STARTING DATA FETCH")
            self.log_message(f"Program: {self.program_choice.get()}")
            self.log_message(f"Program UID: {self.program_uid.get()}")
            self.log_message(f"Level: {self.level_choice.get()}")
            if self.level_choice.get() == "regional":
                self.log_message(
                    f"Region: {self.region_name.get()} ({self.region_uid.get()})"
                )
            self.log_message("=" * 80)

            # Initialize fetcher
            self.fetcher = DHIS2DataFetcher(
                self.base_url.get(), self.username.get(), self.password.get()
            )

            # Fetch orgUnit names
            self.log_message("📋 Fetching orgUnit names...")
            orgunit_names = self.fetcher.fetch_orgunit_names()
            self.log_message(f"✅ Fetched {len(orgunit_names)} orgUnit names")

            # Clear previous data
            self.national_data = []

            # Fetch data based on level
            if self.level_choice.get() == "national":
                self.fetch_national_data(self.program_uid.get(), orgunit_names)
            else:
                self.fetch_regional_data(self.program_uid.get(), orgunit_names)

            # Show success message
            total_patients = (
                sum(len(df) for df in self.national_data) if self.national_data else 0
            )

            self.status_queue.put(
                (
                    "success",
                    f"✅ Data fetch completed successfully!\n\n"
                    f"Total patients: {total_patients}\n"
                    f"Files saved to: {self.output_dir.get()}",
                )
            )

        except Exception as e:
            self.log_message(f"❌ Error in data fetch: {str(e)}")
            self.log_message(traceback.format_exc())
            self.status_queue.put(("error", f"Data fetch error: {str(e)}"))
        finally:
            self.status_queue.put(("finished", None))

    def fetch_national_data(self, program_uid: str, orgunit_names: Dict[str, str]):
        """Fetch data for all regions (national level) - generates ALL regional files"""
        self.log_message("🌍 Fetching NATIONAL data (all regions)")
        self.log_message("📁 Will generate individual CSV files for EACH region")

        # Fetch all regions
        self.log_message("🗺️ Fetching regions...")
        regions = self.fetcher.fetch_all_regions()

        if not regions:
            self.log_message("❌ No regions found")
            return

        total_regions = len(regions)
        self.log_message(
            f"📊 Found {total_regions} regions - will create {total_regions} regional files"
        )

        regional_files_created = 0

        for i, (region_uid, region_name) in enumerate(regions.items(), 1):
            self.log_message(f"\n🏞️ [{i}/{total_regions}] Processing: {region_name}")

            # Fetch program data
            tei_data = self.fetcher.fetch_program_data(
                program_uid, region_uid, "DESCENDANTS", self.page_size.get()
            )

            tei_count = len(tei_data.get("trackedEntityInstances", []))
            self.log_message(f"   📊 Found {tei_count} TEIs")

            if tei_count == 0:
                continue

            # Create events dataframe
            events_df = CSVIntegration.create_events_dataframe(
                tei_data, program_uid, orgunit_names
            )

            # For maternal program only: integrate CSV data
            if program_uid == MATERNAL_PROGRAM_UID and self.csv_data is not None:
                self.log_message("   📁 Integrating CSV data...")
                region_csv = CSVIntegration.filter_csv_data_by_user_access(
                    self.csv_data, "national", region_name, self.facility_to_region_map
                )

                if not region_csv.empty:
                    self.log_message(
                        f"   📊 Found {len(region_csv)} CSV rows for this region"
                    )
                    events_df = CSVIntegration.integrate_maternal_csv_data_for_region(
                        events_df, region_csv, region_name
                    )

            # Transform to patient-level
            patient_df = CSVIntegration.transform_events_to_patient_level(
                events_df, program_uid
            )

            if not patient_df.empty:
                # Post-process
                patient_df = CSVIntegration.post_process_dataframe(patient_df)
                patient_df["region_uid"] = region_uid
                patient_df["region_name"] = region_name
                self.national_data.append(patient_df)

                self.log_message(f"   ✅ Processed {len(patient_df)} patients")

                # Save individual regional file (ONE FILE PER REGION)
                self.save_region_file(patient_df, region_name, program_uid)
                regional_files_created += 1

        # Create combined national file
        if self.national_data:
            self.create_combined_national_file(program_uid)

        self.log_message(
            f"\n📁 Regional files created: {regional_files_created}/{total_regions}"
        )

    def fetch_regional_data(self, program_uid: str, orgunit_names: Dict[str, str]):
        """Fetch data for a single region"""
        region_uid = self.region_uid.get()
        region_name = self.region_name.get()

        if not region_uid or not region_name:
            self.log_message("❌ No region selected")
            return

        self.log_message(f"🏞️ Fetching REGIONAL data for: {region_name}")

        # Fetch program data
        tei_data = self.fetcher.fetch_program_data(
            program_uid, region_uid, "DESCENDANTS", self.page_size.get()
        )

        tei_count = len(tei_data.get("trackedEntityInstances", []))
        self.log_message(f"📊 Found {tei_count} TEIs")

        if tei_count == 0:
            self.log_message("⚠️ No TEIs found for this region")
            return

        # Create events dataframe
        events_df = CSVIntegration.create_events_dataframe(
            tei_data, program_uid, orgunit_names
        )

        # For maternal program only: integrate CSV data
        if program_uid == MATERNAL_PROGRAM_UID and self.csv_data is not None:
            self.log_message("📁 Integrating CSV data...")
            region_csv = CSVIntegration.filter_csv_data_by_user_access(
                self.csv_data, "regional", region_name, self.facility_to_region_map
            )

            if not region_csv.empty:
                self.log_message(f"📊 Found {len(region_csv)} CSV rows for this region")
                events_df = CSVIntegration.integrate_maternal_csv_data_for_region(
                    events_df, region_csv, region_name
                )

        # Transform to patient-level
        patient_df = CSVIntegration.transform_events_to_patient_level(
            events_df, program_uid
        )

        if not patient_df.empty:
            # Post-process
            patient_df = CSVIntegration.post_process_dataframe(patient_df)
            patient_df["region_uid"] = region_uid
            patient_df["region_name"] = region_name
            self.national_data.append(patient_df)

            self.log_message(f"✅ Processed {len(patient_df)} patients")

            # Save regional file
            self.save_region_file(patient_df, region_name, program_uid)

    def save_region_file(
        self, patient_df: pd.DataFrame, region_name: str, program_uid: str
    ):
        """Save individual region file with new naming format: regional_{Region name}_programtype.csv"""
        try:
            # Clean region name for filename
            clean_region = re.sub(r"[^\w\s-]", "", region_name)
            clean_region = re.sub(r"[-\s]+", "_", clean_region)

            # Generate filename
            program_type = (
                "maternal" if program_uid == MATERNAL_PROGRAM_UID else "newborn"
            )
            filename = f"regional_{clean_region}_{program_type}.csv"
            filepath = os.path.join(self.output_dir.get(), filename)

            # Save file (without region columns for cleaner regional files)
            save_df = patient_df.drop(
                columns=["region_uid", "region_name"], errors="ignore"
            )
            save_df.to_csv(filepath, index=False, encoding="utf-8")

            self.log_message(
                f"💾 Saved regional file: {filename} ({len(patient_df)} patients)"
            )

            return filepath

        except Exception as e:
            self.log_message(f"❌ Error saving region file: {str(e)}")
            return None

    def create_combined_national_file(self, program_uid: str):
        """Create combined national file with new naming format: national_programtype.csv"""
        if not self.national_data:
            self.log_message("⚠️ No data to create national file")
            return

        try:
            # Combine all regional data
            combined_df = pd.concat(self.national_data, ignore_index=True)

            # Post-process
            combined_df = CSVIntegration.post_process_dataframe(combined_df)

            # Generate filename
            program_type = (
                "maternal" if program_uid == MATERNAL_PROGRAM_UID else "newborn"
            )
            filename = f"national_{program_type}.csv"
            filepath = os.path.join(self.output_dir.get(), filename)

            # Save file
            combined_df.to_csv(filepath, index=False, encoding="utf-8")

            self.log_message("=" * 80)
            self.log_message(f"🌍 NATIONAL FILE CREATED: {filename}")
            self.log_message(f"📊 Total patients: {len(combined_df)}")
            self.log_message(f"🏞️ Total regions: {combined_df['region_name'].nunique()}")
            self.log_message(
                f"📁 Regional files: {len(self.national_data)} files created"
            )
            self.log_message(f"💾 Saved to: {filepath}")
            self.log_message("=" * 80)

            return filepath

        except Exception as e:
            self.log_message(f"❌ Error creating national file: {str(e)}")
            return None

    def update_status(self):
        """Update status from queue"""
        try:
            while not self.status_queue.empty():
                msg_type, content = self.status_queue.get_nowait()

                if msg_type == "finished":
                    self.is_fetching = False
                    self.fetch_button.config(
                        state=tk.NORMAL, text="🚀 Fetch & Transform Data"
                    )
                    self.progress_bar.stop()
                    self.update_status_label("Ready")

                elif msg_type == "error":
                    self.is_fetching = False
                    self.fetch_button.config(
                        state=tk.NORMAL, text="🚀 Fetch & Transform Data"
                    )
                    self.progress_bar.stop()
                    self.update_status_label("Error occurred")
                    if content:
                        messagebox.showerror("Error", content)

                elif msg_type == "success":
                    self.is_fetching = False
                    self.fetch_button.config(
                        state=tk.NORMAL, text="🚀 Fetch & Transform Data"
                    )
                    self.progress_bar.stop()
                    self.update_status_label("Completed successfully")
                    if content:
                        messagebox.showinfo("Success", content)

                elif msg_type == "message":
                    if content:
                        self.log_message(content)

                elif msg_type == "status":
                    if content:
                        self.update_status_label(content)

        except Exception as e:
            pass

        # Schedule next update
        self.root.after(100, self.update_status)

    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        self.status_text.insert(tk.END, formatted_message + "\n")
        self.status_text.see(tk.END)

        # Also print to console for debugging
        print(formatted_message)

    def update_status_label(self, message: str):
        """Update status label"""
        self.status_label.config(text=f"Status: {message}")

    def clear_log(self):
        """Clear the status log"""
        self.status_text.delete(1.0, tk.END)
        self.log_message("Log cleared")

    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            webbrowser.open(f"file://{output_dir}")
        else:
            messagebox.showwarning(
                "Folder Not Found", f"Output folder does not exist:\n{output_dir}"
            )


# ========== STANDALONE AUTOMATION SCRIPT ==========


def run_automated_pipeline():
    """
    Standalone function to run the automated pipeline
    This can be called from a scheduled task or batch file
    """
    print("=" * 80)
    print("DHIS2 AUTOMATED PIPELINE - STANDALONE")
    print("=" * 80)

    try:
        # Try to import config.py
        import sys

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Add to path for importing config
        sys.path.insert(0, script_dir)

        from config import settings

        # Use credentials from config.py
        config = {
            "base_url": settings.DHIS2_BASE_URL,
            "username": settings.DHIS2_USERNAME,
            "password": settings.DHIS2_PASSWORD,
            "csv_path": "maternal_data_long_format.csv",  # Optional - make sure this file is in the right place
            "output_base_dir": DEFAULT_OUTPUT_DIR,
        }

        print(f"✅ Loaded credentials from config.py")
        print(f"   Base URL: {config['base_url']}")
        print(f"   Username: {config['username']}")
        print(f"   Output directory: {config['output_base_dir']}")
        print(f"   Absolute output path: {os.path.abspath(config['output_base_dir'])}")

        # Create pipeline
        pipeline = AutomatedDHIS2Pipeline(
            base_url=config["base_url"],
            username=config["username"],
            password=config["password"],
            csv_path=config["csv_path"] if os.path.exists(config["csv_path"]) else None,
            output_base_dir=config["output_base_dir"],
        )

        # Run pipeline
        success = pipeline.run_pipeline()

        return success

    except ImportError as e:
        print(f"❌ ERROR: Could not import config.py: {e}")
        print("Please make sure config.py exists in the same directory as this script.")
        return False
    except AttributeError as e:
        print(f"❌ ERROR: Missing setting in config.py: {e}")
        print("Your config.py needs these settings:")
        print("   DHIS2_BASE_URL = 'https://imnid.aau.edu.et/dhis'")
        print("   DHIS2_USERNAME = 'your_username'")
        print("   DHIS2_PASSWORD = 'your_password'")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function - runs GUI by default, can run automation if specified"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--automate":
        # Run automated pipeline
        run_automated_pipeline()
    else:
        # Run GUI
        root = tk.Tk()
        app = DHIS2DataFetcherApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()

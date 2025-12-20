# utils/data_service.py
from typing import Optional, Dict, List, Union, Set, Any
import pandas as pd
import logging
import os
from utils.queries import (
    get_all_programs,
    get_orgunit_uids_for_user,
    get_program_by_uid,
    get_facilities_for_user,
)
from utils.dhis2 import (
    fetch_dhis2_data_for_ous,
    REQUIRED_DATA_ELEMENTS,
    MATERNAL_HEALTH_ELEMENTS,
    NEWBORN_HEALTH_ELEMENTS,
    DATA_ELEMENT_NAMES,
)
from utils.odk_api import fetch_all_forms_as_dataframes, fetch_form_csv, list_forms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def filter_csv_data_by_user_access(csv_df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Filter CSV data based on user's access level and facility permissions.
    """
    if csv_df.empty:
        return csv_df

    user_role = user.get("role", "")

    # Check if CSV has orgUnit_name column
    if "orgUnit_name" not in csv_df.columns:
        logging.warning(
            "CSV data does not have 'orgUnit_name' column - cannot filter by facility"
        )
        return csv_df

    if user_role == "national":
        # National users see all facilities - no filtering needed
        logging.info("🌍 National user - returning ALL CSV data")
        return csv_df

    elif user_role == "regional":
        # Regional users see only facilities in their region
        region_name = user.get("region_name", "")
        if not region_name:
            logging.warning("Regional user has no region_name - cannot filter CSV data")
            return pd.DataFrame()  # Return empty if no region info

        # Get facilities for this regional user
        user_facilities = get_facilities_for_user(user)
        facility_names_in_region = [facility[0] for facility in user_facilities]

        # Filter CSV data to only include facilities in this region
        filtered_csv = csv_df[csv_df["orgUnit_name"].isin(facility_names_in_region)]

        logging.info(
            f"🏞️ Regional user '{region_name}' - filtered CSV to {len(filtered_csv)} rows from {len(facility_names_in_region)} facilities"
        )
        logging.info(f"   📋 Facilities in region: {facility_names_in_region}")
        if not filtered_csv.empty:
            logging.info(
                f"   📊 CSV facilities found: {filtered_csv['orgUnit_name'].unique().tolist()}"
            )

        return filtered_csv

    elif user_role == "facility":
        # Facility users see only their specific facility
        facility_name = user.get("facility_name", "")
        if not facility_name:
            logging.warning(
                "Facility user has no facility_name - cannot filter CSV data"
            )
            return pd.DataFrame()  # Return empty if no facility info

        # Filter CSV data to only include this specific facility
        filtered_csv = csv_df[csv_df["orgUnit_name"] == facility_name]

        logging.info(
            f"🏥 Facility user '{facility_name}' - filtered CSV to {len(filtered_csv)} rows"
        )

        return filtered_csv

    else:
        logging.warning(f"Unknown user role '{user_role}' - returning no CSV data")
        return pd.DataFrame()


def integrate_maternal_csv_data(
    evt_df: pd.DataFrame, user: dict = None
) -> pd.DataFrame:
    """
    Smart integration of maternal CSV data with user-based filtering:
    1. For TEIs with placeholder events in DHIS2 → replace with CSV data
    2. For TEIs with actual events in DHIS2 → check event IDs:
       - If same event ID exists in CSV → replace DHIS2 event with CSV event
       - If different event IDs → keep both (CSV might have additional events)
    3. If both DHIS2 and CSV have placeholder events → prioritize CSV data
    4. Add any new TEIs from CSV that don't exist in DHIS2
    5. FILTER CSV DATA based on user's facility access level
    """
    import os

    # Debug: Show current working directory and file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "maternal_data_long_format.csv")

    logging.info(f"🔍 Looking for CSV at: {csv_path}")
    logging.info(f"🔍 Current directory: {current_dir}")
    logging.info(f"🔍 File exists: {os.path.exists(csv_path)}")

    if not os.path.exists(csv_path):
        logging.info("No maternal CSV file found - using DHIS2 data only")
        return evt_df

    try:
        csv_df = pd.read_csv(csv_path)
        logging.info(
            f"✅ Loaded maternal CSV data: {len(csv_df)} rows, {len(csv_df['tei_id'].unique())} unique TEIs"
        )

        if csv_df.empty:
            logging.info("CSV file is empty - using DHIS2 data only")
            return evt_df

        # ✅ NEW: FILTER CSV DATA BASED ON USER ACCESS LEVEL
        if user:
            csv_df = filter_csv_data_by_user_access(csv_df, user)
            if csv_df.empty:
                logging.info(
                    "No CSV data accessible for this user - using DHIS2 data only"
                )
                return evt_df

        # Get unique TEI IDs from both sources
        csv_tei_ids = (
            set(csv_df["tei_id"].unique()) if "tei_id" in csv_df.columns else set()
        )
        dhis2_tei_ids = (
            set(evt_df["tei_id"].unique()) if "tei_id" in evt_df.columns else set()
        )

        logging.info(
            f"📊 TEI Statistics: DHIS2={len(dhis2_tei_ids)}, CSV={len(csv_tei_ids)}"
        )

        # Identify placeholder vs actual events in both sources
        if "has_actual_event" not in csv_df.columns:
            csv_df["has_actual_event"] = True

        # STRATEGY 1: Handle TEIs that exist in both sources
        common_teis = csv_tei_ids.intersection(dhis2_tei_ids)

        teis_to_remove_from_dhis2 = set()

        for tei_id in common_teis:
            # Get events for this TEI from both sources
            dhis2_tei_events = evt_df[evt_df["tei_id"] == tei_id]
            csv_tei_events = csv_df[csv_df["tei_id"] == tei_id]

            # Check if DHIS2 has ANY placeholder events for this TEI
            dhis2_has_placeholders = any(dhis2_tei_events["has_actual_event"] == False)
            dhis2_has_actual_events = any(dhis2_tei_events["has_actual_event"] == True)

            # Check if CSV has placeholder events for this TEI
            csv_has_placeholders = any(csv_tei_events["has_actual_event"] == False)
            csv_has_actual_events = any(csv_tei_events["has_actual_event"] == True)

            # STRATEGY 1A: Both have placeholders → prioritize CSV
            if dhis2_has_placeholders and csv_has_placeholders:
                logging.info(
                    f"🔄 TEI {tei_id}: Both sources have placeholders - prioritizing CSV data"
                )
                teis_to_remove_from_dhis2.add(tei_id)

            # STRATEGY 1B: DHIS2 has placeholders, CSV has actual events → replace with CSV
            elif dhis2_has_placeholders and csv_has_actual_events:
                logging.info(
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
                    logging.info(
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
                    logging.info(
                        f"📥 TEI {tei_id}: Adding {len(new_events_in_csv)} new events from CSV"
                    )

            # STRATEGY 1D: DHIS2 has actual events, CSV has placeholders → unusual case, log it
            elif dhis2_has_actual_events and csv_has_placeholders:
                logging.warning(
                    f"⚠️ TEI {tei_id}: DHIS2 has actual events but CSV has placeholders - keeping both"
                )

            # STRATEGY 1E: Mixed cases - more detailed logging
            else:
                logging.info(
                    f"🔍 TEI {tei_id}: Mixed case - DHIS2(placeholders:{dhis2_has_placeholders}, actual:{dhis2_has_actual_events}), "
                    f"CSV(placeholders:{csv_has_placeholders}, actual:{csv_has_actual_events})"
                )

        # Remove TEIs that should be completely replaced by CSV
        if teis_to_remove_from_dhis2:
            logging.info(
                f"🗑️ Removing {len(teis_to_remove_from_dhis2)} TEIs from DHIS2 (replaced by CSV)"
            )
            evt_df = evt_df[~evt_df["tei_id"].isin(teis_to_remove_from_dhis2)]

        # STRATEGY 2: Add completely new TEIs from CSV (not in DHIS2 at all)
        new_teis_in_csv = csv_tei_ids - dhis2_tei_ids
        if new_teis_in_csv:
            logging.info(
                f"🆕 Adding {len(new_teis_in_csv)} completely new TEIs from CSV"
            )
            # Log some examples of new TEIs being added
            new_teis_sample = list(new_teis_in_csv)[:5]  # Show first 5 as examples
            for tei_id in new_teis_sample:
                tei_events = csv_df[csv_df["tei_id"] == tei_id]
                has_actual = any(tei_events["has_actual_event"] == True)
                has_placeholder = any(tei_events["has_actual_event"] == False)
                logging.info(
                    f"   ➕ New TEI {tei_id}: events={len(tei_events)}, actual_events={has_actual}, placeholders={has_placeholder}"
                )

        # Add all CSV data (this includes: replaced events, new events for existing TEIs, and new TEIs)
        original_count = len(evt_df)
        evt_df = pd.concat([evt_df, csv_df], ignore_index=True)
        final_count = len(evt_df)

        logging.info(f"✅ FINAL INTEGRATION COMPLETE:")
        logging.info(f"   📈 Before integration: {original_count} rows")
        logging.info(f"   📈 After integration: {final_count} rows")
        logging.info(f"   📈 Net change: {final_count - original_count} rows")
        logging.info(f"   👥 Unique TEIs: {len(evt_df['tei_id'].unique())}")

        # Verify the integration
        integrated_tei_ids = set(evt_df["tei_id"].unique())
        integrated_with_actual_events = len(evt_df[evt_df["has_actual_event"] == True])
        integrated_with_placeholders = len(evt_df[evt_df["has_actual_event"] == False])

        logging.info(f"✅ INTEGRATION VERIFICATION:")
        logging.info(f"   📊 Total TEIs: {len(integrated_tei_ids)}")
        logging.info(f"   📊 Rows with actual events: {integrated_with_actual_events}")
        logging.info(f"   📊 Rows with placeholders: {integrated_with_placeholders}")

    except Exception as e:
        logging.error(f"Failed to integrate CSV data: {e}")
        import traceback

        logging.error(traceback.format_exc())

    return evt_df


def fetch_program_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
    transform_to_patient_level: bool = True,  # ✅ NEW: Add this parameter
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch DHIS2 program data optimized for ALL required KPI data elements.
    Creates placeholder events only for TEI-ProgramStage combinations that have NO events.

    ✅ NEW: Added transform_to_patient_level parameter to control transformation
    """
    if not program_uid:
        logging.warning("No program UID provided.")
        return {}

    program_info = get_program_by_uid(program_uid)
    if not program_info:
        logging.warning(f"Program with UID {program_uid} not found in database.")
        return {}

    # Get program name for logging
    program_name = program_info.get("program_name", "Unknown Program")

    # ✅ SIMPLE CHECK: If program UID is maternal, integrate CSV data
    if program_uid == "aLoraiFNkng":
        logging.info("🔄 Maternal program detected - will integrate CSV data")

    if facility_uids:
        ou_uids = facility_uids
        all_ou_pairs = get_orgunit_uids_for_user(user)
        ou_names = {uid: name for uid, name in all_ou_pairs if uid in facility_uids}
    else:
        ou_pairs = get_orgunit_uids_for_user(user)
        ou_uids = [ou for ou, _ in ou_pairs]
        ou_names = {uid: name for uid, name in ou_pairs}

    if not ou_uids:
        logging.warning("No accessible orgUnits found.")
        return {}

    user_role = user.get("role", "")

    # Use optimized fetch with ALL required data elements
    logging.info(
        f"🚀 OPTIMIZED FETCH for {program_name}: Using {len(REQUIRED_DATA_ELEMENTS)} required data elements"
    )
    logging.info(f"📋 Maternal Health elements: {len(MATERNAL_HEALTH_ELEMENTS)}")
    logging.info(f"📋 Newborn Health elements: {len(NEWBORN_HEALTH_ELEMENTS)}")

    dhis2_data = fetch_dhis2_data_for_ous(
        program_uid, ou_uids, user_role, required_elements=REQUIRED_DATA_ELEMENTS
    )

    patients = dhis2_data.get("patients", [])
    de_dict = dhis2_data.get("dataElements", {})
    ps_dict = dhis2_data.get("programStages", {})
    dhis2_ou_names = dhis2_data.get("orgUnitNames", {})
    optimization_stats = dhis2_data.get("optimization_stats", {})

    final_ou_names = {**dhis2_ou_names, **ou_names}

    if not patients:
        logging.warning(
            f"No patient data found for program {program_info['program_name']}."
        )
        return {"program_info": program_info}

    def map_org_name(uid: str) -> str:
        return final_ou_names.get(uid, "Unknown OrgUnit")

    # TEI DataFrame - ALL TEIs INCLUDED
    tei_df = pd.json_normalize(
        patients,
        record_path=["attributes"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore",
    ).rename(
        columns={"tei_trackedEntityInstance": "tei_id", "tei_orgUnit": "tei_orgUnit"}
    )
    if not tei_df.empty:
        tei_df["orgUnit_name"] = tei_df["tei_orgUnit"].apply(map_org_name)

    # Enrollment DataFrame - ALL enrollments included
    enr_df = pd.json_normalize(
        patients,
        record_path=["enrollments"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore",
    ).rename(
        columns={"tei_trackedEntityInstance": "tei_id", "tei_orgUnit": "tei_orgUnit"}
    )
    if not enr_df.empty:
        enr_df["orgUnit_name"] = enr_df["tei_orgUnit"].apply(map_org_name)

    # Define program stage to data element mapping for BOTH maternal and newborn
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

    NEWBORN_PROGRAM_STAGE_MAPPING = {
        "l39SlVGlQGs": {  # Admission Information
            "data_elements": [
                "UOmhJkyAK6h",  # Date of Admission
                "yxWUMt3sCil",  # Weight on admission
                "T30GbTiVgFR",  # First Reason for Admission
                "OpHw2X58x5i",  # Second Reason for Admission
                "gJH6PkYI6IV",  # Third Reason for Admission
                "aK5txmRYpVX",  # NEW: birth location (inborn/outborn)
            ],
            "program_stage_name": "Admission Information",
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

    # Select the appropriate program stage mapping based on program UID
    if program_uid == "aLoraiFNkng":  # Maternal program
        PROGRAM_STAGE_MAPPING = MATERNAL_PROGRAM_STAGE_MAPPING
        logging.info("📋 Using MATERNAL program stage mapping")
    else:  # Newborn program (or any other)
        PROGRAM_STAGE_MAPPING = NEWBORN_PROGRAM_STAGE_MAPPING
        logging.info("📋 Using NEWBORN program stage mapping")

    # Events DataFrame - CORRECTED APPROACH: Check TEI-ProgramStage combinations
    events_list = []
    required_events_count = 0

    # Track all TEI IDs and their events by program stage
    all_tei_ids = {tei.get("trackedEntityInstance") for tei in patients}

    # Create a mapping of TEI to orgUnit and enrollment info
    tei_info_map = {}
    for tei in patients:
        tei_id = tei.get("trackedEntityInstance")
        tei_org_unit = tei.get("orgUnit")
        enrollment_date = None
        if tei.get("enrollments"):
            enrollment_date = tei["enrollments"][0].get("enrollmentDate")

        tei_info_map[tei_id] = {
            "orgUnit": tei_org_unit,
            "enrollmentDate": enrollment_date,
        }

    # Track which TEI-ProgramStage combinations have actual events
    tei_program_stage_events = set()

    # Process actual events from DHIS2 - RETURN ALL REQUIRED ELEMENTS REGARDLESS OF VALUE
    for tei in patients:
        tei_id = tei.get("trackedEntityInstance")
        tei_org_unit = tei.get("orgUnit")

        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event_orgUnit = (
                    event.get("orgUnit") or enrollment.get("orgUnit") or tei_org_unit
                )
                event_date = event.get("eventDate")
                program_stage_uid = event.get("programStage")

                # Mark that this TEI has an event for this program stage UID
                tei_program_stage_events.add((tei_id, program_stage_uid))

                program_stage_name = event.get(
                    "programStageName",
                    ps_dict.get(program_stage_uid, program_stage_uid),
                )

                # Get all required data elements for this program stage
                stage_required_elements = PROGRAM_STAGE_MAPPING.get(
                    program_stage_uid, {}
                ).get("data_elements", [])

                # Process ALL required data elements for this program stage
                for data_element_uid in stage_required_elements:
                    # Find the data value for this element (if it exists)
                    data_value = ""
                    for dv in event.get("dataValues", []):
                        if dv.get("dataElement") == data_element_uid:
                            data_value = dv.get("value", "")
                            break

                    data_element_name = DATA_ELEMENT_NAMES.get(
                        data_element_uid,
                        de_dict.get(data_element_uid, data_element_uid),
                    )

                    event_data = {
                        "tei_id": tei_id,
                        "event": event.get("event"),
                        "programStage_uid": program_stage_uid,
                        "programStageName": program_stage_name,
                        "orgUnit": event_orgUnit,
                        "orgUnit_name": map_org_name(event_orgUnit),
                        "eventDate": event_date,
                        "dataElement_uid": data_element_uid,
                        "dataElementName": data_element_name,
                        "value": data_value,  # Will be empty if no value recorded
                        "has_actual_event": True,
                    }
                    events_list.append(event_data)
                    required_events_count += 1

    # Create placeholder events for TEI-ProgramStage combinations that have NO events
    placeholder_count = 0
    for tei_id in all_tei_ids:
        tei_info = tei_info_map.get(tei_id, {})

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

                # Create one placeholder row for EACH data element in this program stage
                for data_element_uid in stage_info["data_elements"]:
                    data_element_name = DATA_ELEMENT_NAMES.get(
                        data_element_uid,
                        de_dict.get(data_element_uid, data_element_uid),
                    )

                    placeholder_event = {
                        "tei_id": tei_id,
                        "event": f"placeholder_{tei_id}_{program_stage_uid}",
                        "programStage_uid": program_stage_uid,
                        "programStageName": stage_info["program_stage_name"],
                        "orgUnit": tei_info.get("orgUnit"),
                        "orgUnit_name": map_org_name(tei_info.get("orgUnit")),
                        "eventDate": event_date,
                        "dataElement_uid": data_element_uid,
                        "dataElementName": data_element_name,
                        "value": "",  # Empty value for missing data
                        "has_actual_event": False,
                    }
                    events_list.append(placeholder_event)
                    placeholder_count += 1

    evt_df = pd.DataFrame(events_list)

    # Log statistics about events and placeholders
    tei_program_stage_combinations = len(all_tei_ids) * len(PROGRAM_STAGE_MAPPING)
    actual_tei_program_stage_combinations = len(tei_program_stage_events)

    logging.info(f"📊 EVENT STATISTICS for {program_name}:")
    logging.info(f"   👥 Total TEIs: {len(all_tei_ids)}")
    logging.info(f"   🏥 Total program stages: {len(PROGRAM_STAGE_MAPPING)}")
    logging.info(
        f"   📋 Possible TEI-ProgramStage combinations: {tei_program_stage_combinations}"
    )
    logging.info(
        f"   ✅ Actual TEI-ProgramStage combinations with events: {actual_tei_program_stage_combinations}"
    )
    logging.info(
        f"   📝 Placeholder TEI-ProgramStage combinations: {tei_program_stage_combinations - actual_tei_program_stage_combinations}"
    )
    logging.info(f"   📈 Required data events collected: {required_events_count}")
    logging.info(f"   📝 Placeholder events created: {placeholder_count}")

    # ✅ SMART CSV INTEGRATION FOR MATERNAL PROGRAM ONLY
    if program_uid == "aLoraiFNkng":
        logging.info("🔄 Integrating CSV data for maternal program")
        evt_df = integrate_maternal_csv_data(evt_df, user)  # ✅ Pass user parameter

    # Handle period labeling
    if not evt_df.empty and "eventDate" in evt_df.columns:
        evt_df["event_date"] = pd.to_datetime(evt_df["eventDate"], errors="coerce")
        has_date_mask = evt_df["event_date"].notna()

        if period_label == "Daily":
            evt_df.loc[has_date_mask, "period"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.date.astype(str)
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y-%m-%d")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y%m%d")
        elif period_label == "Monthly":
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("M").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%b %Y")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y%m")
        elif period_label == "Quarterly":
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Q").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Q").astype(str)
            )
            evt_df.loc[has_date_mask, "period_sort"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.year.astype(str)
                + "Q"
                + evt_df.loc[has_date_mask, "event_date"].dt.quarter.astype(str)
            )
        else:
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Y").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.year.astype(str)

    # Convert enrollment dates
    if not enr_df.empty and "enrollmentDate" in enr_df.columns:
        enr_df["enrollmentDate"] = pd.to_datetime(
            enr_df["enrollmentDate"], errors="coerce"
        )

    # ✅ NEW: TRANSFORM TO PATIENT-LEVEL FORMAT IF REQUESTED
    patient_df = pd.DataFrame()
    if transform_to_patient_level and not evt_df.empty:
        logging.info(f"🔄 Transforming {program_name} events to patient-level format")
        patient_df = transform_events_to_patient_level(evt_df, program_uid)
        if not patient_df.empty:
            logging.info(
                f"✅ Patient-level transformation complete: {len(patient_df)} patients, {len(patient_df.columns)} columns"
            )

            # ✅ DEBUG: Save transformed data to CSV
            try:
                import os
                from datetime import datetime

                # Create data_exports directory if it doesn't exist
                os.makedirs("data_exports", exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                program_type = "maternal" if program_uid == "aLoraiFNkng" else "newborn"
                filename = f"transformed_{program_type}_{timestamp}.csv"
                filepath = os.path.join("data_exports", filename)

                patient_df.to_csv(filepath, index=False, encoding="utf-8")
                logging.info(f"💾 Saved transformed data to: {filepath}")
                logging.info(f"   File size: {os.path.getsize(filepath) / 1024:.1f} KB")
            except Exception as e:
                logging.warning(f"⚠️ Could not save transformed data: {e}")
        else:
            logging.warning("⚠️ Patient-level transformation returned empty DataFrame")

    # Data verification
    if not evt_df.empty:
        # Count actual vs placeholder events
        actual_events = len(evt_df[evt_df["has_actual_event"] == True])
        placeholder_events = len(evt_df[evt_df["has_actual_event"] == False])

        logging.info(f"✅ FINAL DATA STRUCTURE for {program_name}:")
        logging.info(f"   📊 Total events: {len(evt_df)}")
        logging.info(f"   ✅ Actual events: {actual_events}")
        logging.info(f"   📝 Placeholder events: {placeholder_events}")
        logging.info(f"   👥 Unique TEIs: {len(evt_df['tei_id'].unique())}")

    # Build result dictionary
    result = {
        "program_info": program_info,
        "raw_json": patients,
        "tei": tei_df,
        "enrollments": enr_df,
        "events": evt_df,
        "optimization_stats": optimization_stats,
    }

    # Add patient-level data if transformed
    if not patient_df.empty:
        result["patients"] = patient_df  # ✅ NEW: Add patient-level data
        logging.info(f"📊 Added patient-level data to result: {len(patient_df)} rows")
    else:
        logging.info(f"📊 No patient-level data available for {program_name}")

    return result


def fetch_odk_data_for_user(
    user: dict, form_id: str = None
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Fetch ODK form data filtered by user's access level."""
    odk_data = {}

    try:
        if form_id:
            df = fetch_form_csv(form_id)
            if not df.empty and user:
                from utils.odk_api import filter_by_region

                user_role = user.get("role", "")
                if user_role == "regional" and user.get("region_id"):
                    df = filter_by_region(df, user.get("region_id"))
                    logging.info(
                        f"Applied regional filtering for form {form_id}, region_id: {user.get('region_id')}"
                    )
                elif user_role == "national":
                    logging.info(
                        f"✅ National user - NO filtering for form {form_id} - returning ALL data"
                    )

            if not df.empty:
                odk_data[form_id] = df
                logging.info(
                    f"Retrieved ODK form '{form_id}' with {len(df)} records and {len(df.columns)} columns"
                )
            else:
                logging.warning(f"No data found for ODK form '{form_id}'")
        else:
            form_dfs = fetch_all_forms_as_dataframes(user)
            odk_data.update(form_dfs)

            user_role = user.get("role", "unknown") if user else "none"
            logging.info(
                f"✅ Retrieved {len(form_dfs)} ODK forms for user role: {user_role}"
            )

    except Exception as e:
        logging.error(f"Error fetching ODK data: {e}")

    return {"odk_forms": odk_data}


def fetch_combined_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
    odk_form_id: str = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch both DHIS2 program data and ODK form data for a user.
    """
    result = {}

    if program_uid:
        dhis2_data = fetch_program_data_for_user(
            user, program_uid, facility_uids, period_label
        )
        result.update(dhis2_data)

    odk_data = fetch_odk_data_for_user(user, odk_form_id)
    result.update(odk_data)

    return result


def list_available_odk_forms() -> List[dict]:
    """List all available ODK forms in the project."""
    try:
        forms = list_forms()
        return forms
    except Exception as e:
        logging.error(f"Error listing ODK forms: {e}")
        return []


# ✅ TRANSFORMATION FUNCTIONS START HERE


def transform_events_to_patient_level(
    events_df: pd.DataFrame, program_uid: str
) -> pd.DataFrame:
    """
    Transform events DataFrame from long format (one row per data element)
    to wide format (one row per patient) with program stage information in column names.

    This creates a much more efficient dataset for analysis.

    Args:
        events_df: DataFrame in the current long format
        program_uid: Program UID to determine program type

    Returns:
        DataFrame in wide format (one row per patient)
    """
    if events_df.empty:
        logging.warning("Empty events DataFrame provided for transformation")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original
    df = events_df.copy()

    # Determine program type based on UID
    is_maternal = program_uid == "aLoraiFNkng"
    program_type = "maternal" if is_maternal else "newborn"

    logging.info(
        f"🔄 Transforming {program_type.upper()} events to patient-level format"
    )
    logging.info(
        f"   Input: {len(df)} rows from {df['tei_id'].nunique()} unique patients"
    )
    logging.info(f"   Found {df['programStage_uid'].nunique()} unique program stages")
    logging.info(f"   Found {df['dataElement_uid'].nunique()} unique data elements")

    # Step 1: Create base patient information
    if "orgUnit" in df.columns and "orgUnit_name" in df.columns:
        patient_base = df[["tei_id", "orgUnit", "orgUnit_name"]].drop_duplicates()
    else:
        # Fallback if columns don't exist
        patient_base = df[["tei_id"]].drop_duplicates()
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

    # Step 2: Get all unique program stages for this data
    program_stages = df[["programStage_uid", "programStageName"]].drop_duplicates()

    # Step 3: Process each program stage separately
    all_stage_data = []

    for _, stage_row in program_stages.iterrows():
        program_stage_uid = stage_row["programStage_uid"]
        program_stage_name = stage_row["programStageName"]

        # Filter data for this program stage
        stage_data = df[df["programStage_uid"] == program_stage_uid].copy()

        if stage_data.empty:
            continue

        logging.info(
            f"   Processing program stage: {program_stage_name} ({program_stage_uid})"
        )

        # Group by tei_id - each group represents one patient in this program stage
        stage_groups = stage_data.groupby("tei_id")

        stage_patient_rows = []

        for tei_id, group in stage_groups:
            # Start with basic patient info
            patient_row = {"tei_id": tei_id}

            # Get the first event row for metadata (if there are multiple events for same stage)
            if not group.empty:
                first_event = group.iloc[0]

                # Add event metadata with program stage suffix
                # Use UID suffix for technical columns
                if "event" in first_event:
                    patient_row[f"event_{program_stage_uid}"] = first_event.get(
                        "event", ""
                    )
                if "eventDate" in first_event:
                    patient_row[f"eventDate_{program_stage_uid}"] = first_event.get(
                        "eventDate", ""
                    )
                if "has_actual_event" in first_event:
                    patient_row[f"has_actual_event_{program_stage_uid}"] = (
                        first_event.get("has_actual_event", "")
                    )

                # Use name suffix for display columns
                if "event" in first_event:
                    patient_row[f"event_{program_stage_name}"] = first_event.get(
                        "event", ""
                    )
                if "eventDate" in first_event:
                    patient_row[f"eventDate_{program_stage_name}"] = first_event.get(
                        "eventDate", ""
                    )
                if "has_actual_event" in first_event:
                    patient_row[f"has_actual_event_{program_stage_name}"] = (
                        first_event.get("has_actual_event", "")
                    )

                # Add date-related columns with program stage name suffix
                for date_col in [
                    "event_date",
                    "period",
                    "period_display",
                    "period_sort",
                ]:
                    if date_col in first_event:
                        patient_row[f"{date_col}_{program_stage_name}"] = first_event[
                            date_col
                        ]

            # Add data elements - create columns for each data element
            # We need to handle the case where a patient might have multiple events in the same program stage
            # For now, we'll take the first non-empty value
            for data_element_uid in group["dataElement_uid"].unique():
                # Get all values for this data element
                element_values = group[group["dataElement_uid"] == data_element_uid][
                    "value"
                ]
                # Take the first non-empty value if available
                value = next(
                    (v for v in element_values if pd.notna(v) and str(v).strip() != ""),
                    "",
                )

                # Get the data element name
                data_element_rows = group[group["dataElement_uid"] == data_element_uid]
                if not data_element_rows.empty:
                    data_element_name = data_element_rows["dataElementName"].iloc[0]

                    # Create column with UID + program stage UID
                    uid_col_name = f"{data_element_uid}_{program_stage_uid}"
                    patient_row[uid_col_name] = value

                    # Create column with name + program stage name
                    # Clean the column name to avoid special characters
                    clean_name = (
                        str(data_element_name)
                        .replace("(", "")
                        .replace(")", "")
                        .replace(":", "")
                        .replace("/", "_")
                        .replace(" ", "_")
                        .replace(".", "")
                    )
                    # Remove multiple underscores
                    while "__" in clean_name:
                        clean_name = clean_name.replace("__", "_")
                    # Remove trailing/leading underscores
                    clean_name = clean_name.strip("_")

                    name_col_name = f"{clean_name}_{program_stage_name}"
                    # Clean the program stage name too
                    name_col_name = (
                        name_col_name.replace(" ", "_")
                        .replace("-", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )
                    while "__" in name_col_name:
                        name_col_name = name_col_name.replace("__", "_")
                    name_col_name = name_col_name.strip("_")

                    patient_row[name_col_name] = value

            stage_patient_rows.append(patient_row)

        # Create DataFrame for this program stage
        if stage_patient_rows:
            stage_df = pd.DataFrame(stage_patient_rows)
            all_stage_data.append(stage_df)

    # Step 4: Merge all stage data with patient base
    if not all_stage_data:
        logging.warning("No program stage data found after transformation")
        return patient_base

    # Start with patient base and merge all program stage data
    patient_df = patient_base
    for stage_df in all_stage_data:
        patient_df = patient_df.merge(stage_df, on="tei_id", how="left")

    # Fill NaN values with empty string for better readability
    patient_df = patient_df.fillna("")

    # Add program type indicator
    patient_df["program_type"] = program_type

    # Log transformation statistics
    logging.info(f"✅ Transformation complete:")
    logging.info(
        f"   📊 Output: {len(patient_df)} patients, {len(patient_df.columns)} columns"
    )
    if len(df) > 0 and len(patient_df) > 0:
        reduction_pct = (1 - len(patient_df) / len(df)) * 100
        logging.info(
            f"   📈 Data reduction: {len(df)} → {len(patient_df)} rows ({reduction_pct:.1f}% reduction)"
        )

    # Show some sample column names
    sample_cols = [
        col
        for col in patient_df.columns
        if col not in ["tei_id", "orgUnit", "orgUnit_name", "program_type"]
    ]
    if len(sample_cols) > 0:
        logging.info(f"   📋 Sample columns ({len(sample_cols)} total):")
        for i, col in enumerate(sample_cols[:10]):  # Show first 10
            logging.info(f"      {i+1}. {col}")
        if len(sample_cols) > 10:
            logging.info(f"      ... and {len(sample_cols) - 10} more columns")

    return patient_df


def get_newborn_program_uid() -> str:
    """
    Get the newborn program UID from the database.
    Returns the UID for 'Newborn Care Form' program.
    """
    programs = get_all_programs()
    for program in programs:
        if program.get("program_name") == "Newborn Care Form":
            return program.get("program_uid", "")

    # If not found, try to find any program that's not maternal
    for program in programs:
        if program.get("program_uid") != "aLoraiFNkng":
            return program.get("program_uid", "")

    return ""


def get_patient_level_summary(patient_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics from patient-level DataFrame.

    Args:
        patient_df: Patient-level DataFrame

    Returns:
        Dictionary with summary statistics
    """
    if patient_df.empty:
        return {"error": "Empty DataFrame"}

    summary = {
        "total_patients": len(patient_df),
        "total_columns": len(patient_df.columns),
        "orgunits_count": (
            patient_df["orgUnit_name"].nunique()
            if "orgUnit_name" in patient_df.columns
            else 0
        ),
        "program_type": (
            patient_df["program_type"].iloc[0]
            if "program_type" in patient_df.columns
            else "unknown"
        ),
        "column_categories": {},
    }

    # Categorize columns
    columns = list(patient_df.columns)

    for col in columns:
        if col in ["tei_id", "orgUnit", "orgUnit_name", "program_type"]:
            summary["column_categories"]["basic_info"] = (
                summary["column_categories"].get("basic_info", 0) + 1
            )
        elif (
            col.startswith("event_")
            or col.startswith("eventDate_")
            or col.startswith("has_actual_event_")
        ):
            summary["column_categories"]["event_metadata"] = (
                summary["column_categories"].get("event_metadata", 0) + 1
            )
        elif "_" in col and any(
            x in col
            for x in ["event_date_", "period_", "period_display_", "period_sort_"]
        ):
            summary["column_categories"]["date_info"] = (
                summary["column_categories"].get("date_info", 0) + 1
            )
        elif "_" in col and len(col.split("_")) >= 2:
            # Likely a data element column
            summary["column_categories"]["data_elements"] = (
                summary["column_categories"].get("data_elements", 0) + 1
            )
        else:
            summary["column_categories"]["other"] = (
                summary["column_categories"].get("other", 0) + 1
            )

    return summary


def optimize_patient_dataframe(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize the patient-level DataFrame for better performance.

    Args:
        patient_df: Patient-level DataFrame

    Returns:
        Optimized DataFrame
    """
    if patient_df.empty:
        return patient_df

    df = patient_df.copy()

    # Reorder columns: basic info first, then metadata, then data elements
    basic_cols = ["tei_id", "orgUnit", "orgUnit_name", "program_type"]
    event_cols = [
        col
        for col in df.columns
        if col.startswith(("event_", "eventDate_", "has_actual_event_"))
    ]
    date_cols = [
        col
        for col in df.columns
        if any(
            x in col
            for x in ["event_date_", "period_", "period_display_", "period_sort_"]
        )
    ]
    data_cols = [
        col for col in df.columns if col not in basic_cols + event_cols + date_cols
    ]

    # Sort each category
    event_cols.sort()
    date_cols.sort()
    data_cols.sort()

    # Reorder DataFrame
    ordered_cols = basic_cols + event_cols + date_cols + data_cols
    df = df[ordered_cols]

    return df


def create_data_exports_directory():
    """Create the data_exports directory if it doesn't exist."""
    import os

    os.makedirs("data_exports", exist_ok=True)
    logging.info("📁 Created/verified data_exports directory")


# Initialize data exports directory
create_data_exports_directory()

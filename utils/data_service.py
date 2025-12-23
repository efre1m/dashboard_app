# utils/data_service.py
from typing import Optional, Dict, List, Union, Set, Any
import pandas as pd
import logging
import os
import glob
import re
from utils.queries import (
    get_all_programs,
    get_orgunit_uids_for_user,
    get_program_by_uid,
    get_facilities_for_user,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_data_directory(program_type: str = "maternal") -> str:
    """
    Get the directory path for the specific program type.

    Args:
        program_type: "maternal" or "newborn"

    Returns:
        Path to the data directory
    """
    base_dir = os.path.join(os.path.dirname(__file__), "imnid")

    if program_type == "maternal":
        return os.path.join(base_dir, "maternal")
    elif program_type == "newborn":
        return os.path.join(base_dir, "newborn")
    else:
        return base_dir


def normalize_region_name(region_name: str) -> str:
    """
    Normalize region name to match file naming convention.

    Args:
        region_name: Region name from user data

    Returns:
        Normalized region name for file matching
    """
    if not region_name:
        return ""

    normalized = region_name.lower().strip()
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = re.sub(r"[\s-]+", "_", normalized)

    return normalized


def get_user_csv_file(user: dict, program_type: str = "maternal") -> str:
    """
    Determine which CSV file to load based on user role and program type.

    Args:
        user: User dictionary with role, region_name, facility_name
        program_type: "maternal" or "newborn"

    Returns:
        Path to the appropriate CSV file or None if not found
    """
    user_role = user.get("role", "")
    region_name = user.get("region_name", "")

    data_dir = get_data_directory(program_type)

    if not os.path.exists(data_dir):
        logging.error(f"❌ Directory does not exist: {data_dir}")
        return None

    logging.info(f"🔍 Looking for {program_type} data in: {data_dir}")

    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        logging.error(f"❌ No CSV files found in directory: {data_dir}")
        return None

    logging.info(f"   Available files: {[os.path.basename(f) for f in all_files]}")

    if user_role == "national":
        exact_file = os.path.join(data_dir, f"national_{program_type}.csv")
        if os.path.exists(exact_file):
            logging.info(f"✅ Found exact national file: national_{program_type}.csv")
            return exact_file

        national_files = [
            f for f in all_files if "national" in os.path.basename(f).lower()
        ]
        if national_files:
            national_files.sort(key=os.path.getmtime, reverse=True)
            selected_file = national_files[0]
            logging.info(f"✅ Found national file: {os.path.basename(selected_file)}")
            return selected_file

        all_files.sort(key=os.path.getmtime, reverse=True)
        return all_files[0]

    elif user_role == "regional":
        if not region_name:
            logging.error("Regional user has no region_name - cannot determine file")
            return None

        normalized_region = normalize_region_name(region_name)

        logging.info(
            f"   Looking for region: {region_name} (normalized: {normalized_region})"
        )

        exact_pattern = f"regional_{normalized_region}_{program_type}.csv"
        exact_file = os.path.join(data_dir, exact_pattern)

        if os.path.exists(exact_file):
            logging.info(f"✅ Found exact regional file: {exact_pattern}")
            return exact_file

        region_files = []
        for filepath in all_files:
            filename = os.path.basename(filepath).lower()
            if normalized_region in filename and program_type in filename:
                region_files.append(filepath)

        if region_files:
            region_files.sort(key=os.path.getmtime, reverse=True)
            selected_file = region_files[0]
            logging.info(f"✅ Found regional file: {os.path.basename(selected_file)}")
            return selected_file

        region_parts = normalized_region.split("_")
        for i in range(len(region_parts), 0, -1):
            partial_region = "_".join(region_parts[:i])
            for filepath in all_files:
                filename = os.path.basename(filepath).lower()
                if partial_region in filename and program_type in filename:
                    logging.info(
                        f"✅ Found file with partial region match: {os.path.basename(filepath)}"
                    )
                    return filepath

        regional_files = [
            f for f in all_files if "regional" in os.path.basename(f).lower()
        ]
        if regional_files:
            regional_files.sort(key=os.path.getmtime, reverse=True)
            selected_file = regional_files[0]
            logging.warning(
                f"⚠️ Using generic regional file: {os.path.basename(selected_file)}"
            )
            return selected_file

        all_files.sort(key=os.path.getmtime, reverse=True)
        return all_files[0]

    elif user_role == "facility":
        exact_file = os.path.join(data_dir, f"national_{program_type}.csv")
        if os.path.exists(exact_file):
            logging.info(
                f"✅ Found exact national file for facility user: national_{program_type}.csv"
            )
            return exact_file

        national_files = [
            f for f in all_files if "national" in os.path.basename(f).lower()
        ]
        if national_files:
            national_files.sort(key=os.path.getmtime, reverse=True)
            selected_file = national_files[0]
            logging.info(
                f"✅ Found national file for facility user: {os.path.basename(selected_file)}"
            )
            return selected_file

        all_files.sort(key=os.path.getmtime, reverse=True)
        return all_files[0]

    logging.warning(f"⚠️ Unknown user role '{user_role}', using first available file")
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[0]


def filter_patient_data_by_user(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Filter patient-level DataFrame based on user's access level.

    Args:
        df: Patient-level DataFrame
        user: User dictionary with role, region_name, facility_name

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    user_role = user.get("role", "")

    if user_role == "national":
        logging.info("🌍 National user - returning ALL data")
        return df

    elif user_role == "regional":
        region_name = user.get("region_name", "")
        if not region_name:
            logging.warning("Regional user has no region_name - cannot filter data")
            return pd.DataFrame()

        user_facilities = get_facilities_for_user(user)
        facility_names_in_region = [facility[0] for facility in user_facilities]

        from utils.queries import get_facility_mapping_for_user

        facility_mapping = get_facility_mapping_for_user(user)

        facility_uids_in_region = []
        for facility_name in facility_names_in_region:
            if facility_name in facility_mapping:
                facility_uids_in_region.append(facility_mapping[facility_name])

        if "orgUnit" in df.columns and facility_uids_in_region:
            filtered_df = df[df["orgUnit"].isin(facility_uids_in_region)]
        elif "orgUnit_name" in df.columns and facility_names_in_region:
            filtered_df = df[df["orgUnit_name"].isin(facility_names_in_region)]
        else:
            logging.warning(
                "DataFrame has no orgUnit or orgUnit_name column - cannot filter"
            )
            return df

        logging.info(
            f"🏞️ Regional user '{region_name}' - filtered to {len(filtered_df)} patients from {len(facility_uids_in_region)} facilities"
        )
        return filtered_df

    elif user_role == "facility":
        facility_name = user.get("facility_name", "")
        if not facility_name:
            logging.warning("Facility user has no facility_name - cannot filter data")
            return pd.DataFrame()

        from utils.queries import get_facility_mapping_for_user

        facility_mapping = get_facility_mapping_for_user(user)
        facility_uid = facility_mapping.get(facility_name)

        if facility_uid and "orgUnit" in df.columns:
            filtered_df = df[df["orgUnit"] == facility_uid]
        elif "orgUnit_name" in df.columns:
            filtered_df = df[df["orgUnit_name"] == facility_name]
        else:
            logging.warning(
                "DataFrame has no orgUnit or orgUnit_name column - cannot filter"
            )
            return df

        logging.info(
            f"🏥 Facility user '{facility_name}' - filtered to {len(filtered_df)} patients"
        )
        return filtered_df

    else:
        logging.warning(f"Unknown user role '{user_role}' - returning no data")
        return pd.DataFrame()


def transform_patient_to_events(df: pd.DataFrame, program_type: str) -> pd.DataFrame:
    """
    Transform patient-level data to events DataFrame format for compatibility.
    This creates the 'events' DataFrame that regional.py expects.
    """
    if df.empty:
        return pd.DataFrame()

    events_list = []

    # Define program stage mappings
    program_stages = {
        "maternal": {
            "delivery_summary": "mdw5BoS50mb",
            "postpartum_care": "VpBHRE7FlJL",
            "discharge_summary": "DLVsIxjhwMj",
            "instrumental_delivery": "bwk9bBfYcsD",
        },
        "newborn": {
            "microbiology_and_labs": "aCrttmnx7FI",
            "admission_information": "l39SlVGlQGs",
            "observations_and_nursing_care_1": "j0HI2eJjvbj",
            "interventions": "ed8ErpgTCwx",
            "observations_and_nursing_care_2": "VsVlpG1V2ub",
            "discharge_and_final_diagnosis": "TOicTEwzSGj",
        },
    }

    stages = program_stages.get(program_type, {})

    for idx, row in df.iterrows():
        tei_id = row.get("tei_id", f"patient_{idx}")
        org_unit = row.get("orgUnit", "")
        org_unit_name = row.get("orgUnit_name", "")

        # Process each program stage
        for stage_key, stage_uid in stages.items():
            # Check if this stage exists in the data
            event_col = f"event_{stage_key}"
            event_date_col = f"event_date_{stage_key}"
            period_col = f"period_{stage_key}"
            period_display_col = f"period_display_{stage_key}"

            if event_col in row and pd.notna(row.get(event_col)):
                event_id = row[event_col]
                event_date = row.get(event_date_col)
                period = row.get(period_col, "")
                period_display = row.get(period_display_col, "")

                # Get all data elements for this stage
                data_element_prefixes = [
                    col.replace(f"{stage_key}_", "")
                    for col in df.columns
                    if col.startswith(f"{stage_key}_")
                    and col
                    not in [event_col, event_date_col, period_col, period_display_col]
                ]

                # Create events for each data element
                for data_element in data_element_prefixes:
                    value_col = f"{stage_key}_{data_element}"
                    if value_col in row:
                        event_data = {
                            "tei_id": tei_id,
                            "event": event_id,
                            "programStage_uid": stage_uid,
                            "programStageName": stage_key.replace("_", " ").title(),
                            "orgUnit": org_unit,
                            "orgUnit_name": org_unit_name,
                            "eventDate": event_date,
                            "dataElement_uid": data_element,  # Simplified
                            "dataElementName": data_element.replace("_", " ").title(),
                            "value": row[value_col],
                            "has_actual_event": True,
                            "period": period,
                            "period_display": period_display,
                            "period_sort": row.get(f"period_sort_{stage_key}", ""),
                        }
                        events_list.append(event_data)

    events_df = pd.DataFrame(events_list)

    # Add event_date column for compatibility with dash_co functions
    if not events_df.empty and "eventDate" in events_df.columns:
        events_df["event_date"] = pd.to_datetime(
            events_df["eventDate"], errors="coerce"
        )

    logging.info(f"✅ Transformed {len(df)} patients to {len(events_df)} events")
    return events_df


def load_patient_data_for_user(
    user: dict, program_type: str = "maternal"
) -> pd.DataFrame:
    """
    Load pre-fetched, integrated, and transformed patient-level CSV data for user.

    Args:
        user: User dictionary with role, region_name, facility_name
        program_type: "maternal" or "newborn"

    Returns:
        Patient-level DataFrame filtered for user access
    """
    csv_file = get_user_csv_file(user, program_type)

    if not csv_file or not os.path.exists(csv_file):
        data_dir = get_data_directory(program_type)
        logging.error(f"❌ CSV file not found for {program_type} program")
        logging.error(f"   User role: {user.get('role')}")
        logging.error(f"   User region: {user.get('region_name', 'N/A')}")
        logging.error(f"   Expected in: {data_dir}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
        logging.info(
            f"✅ Loaded {program_type} data from: {os.path.basename(csv_file)}"
        )
        logging.info(f"   📊 Total patients: {len(df)}")
        logging.info(f"   📋 Columns: {len(df.columns)}")

        if "program_type" not in df.columns:
            df["program_type"] = program_type

        if "tei_id" not in df.columns:
            id_cols = [
                col for col in df.columns if "id" in col.lower() or "tei" in col.lower()
            ]
            if id_cols:
                df = df.rename(columns={id_cols[0]: "tei_id"})
                logging.info(f"   🔄 Renamed column '{id_cols[0]}' to 'tei_id'")
            else:
                df["tei_id"] = [f"patient_{i}" for i in range(len(df))]
                logging.info(f"   📝 Created placeholder tei_id column")

        filtered_df = filter_patient_data_by_user(df, user)

        if filtered_df.empty:
            logging.warning(
                f"⚠️ No data accessible for user {user.get('username')} after filtering"
            )
            logging.warning(f"   Original data had {len(df)} patients")
        else:
            logging.info(f"✅ Filtered data for user: {len(filtered_df)} patients")

        return filtered_df

    except Exception as e:
        logging.error(f"❌ Failed to load CSV data from {csv_file}: {e}")
        return pd.DataFrame()


def fetch_program_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
    transform_to_patient_level: bool = True,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Load pre-fetched program data for user from CSV files.

    Args:
        user: User dictionary
        program_uid: Program UID (helps identify program type)
        facility_uids: Facility UIDs (unused for CSV loading)
        period_label: Period label for display
        transform_to_patient_level: Always True - we only have patient-level data

    Returns:
        Dictionary with patient-level data and program info
    """
    program_info = get_program_by_uid(program_uid) if program_uid else {}
    program_name = program_info.get("program_name", "Unknown Program")

    # Map program name/UID to program type
    if (
        program_uid == "aLoraiFNkng"
        or "Maternal" in program_name
        or "maternal" in program_name.lower()
    ):
        program_type = "maternal"
    elif (
        "Newborn" in program_name
        or "newborn" in program_name.lower()
        or program_name == "Newborn Care Form"
    ):
        program_type = "newborn"
    else:
        if program_uid and program_uid == "aLoraiFNkng":
            program_type = "maternal"
        else:
            program_type = "newborn"

    logging.info(f"📂 Loading {program_type} program data")
    logging.info(f"   Program UID: {program_uid}")
    logging.info(f"   Program name: {program_name}")
    logging.info(f"   User role: {user.get('role')}")
    logging.info(f"   User region: {user.get('region_name', 'N/A')}")

    # Load patient-level data from CSV
    patient_df = load_patient_data_for_user(user, program_type)

    # ✅ TRANSFORM to events DataFrame for compatibility
    events_df = transform_patient_to_events(patient_df, program_type)

    # Create empty dataframes for compatibility
    tei_df = pd.DataFrame()
    enr_df = pd.DataFrame()

    # Create minimal program info if not found
    if not program_info:
        program_info = {
            "program_uid": program_uid or "",
            "program_name": f"{program_type.capitalize()} Program",
            "program_type": program_type,
        }

    # Build result dictionary
    result = {
        "program_info": program_info,
        "raw_json": [],
        "tei": tei_df,
        "enrollments": enr_df,
        "events": events_df,  # ✅ TRANSFORMED events DataFrame
        "patients": patient_df,  # Original patient-level data
        "optimization_stats": {
            "data_source": "pre_fetched_csv",
            "program_type": program_type,
            "patient_count": len(patient_df),
            "events_count": len(events_df),
        },
    }

    if not patient_df.empty:
        logging.info(
            f"✅ Successfully loaded {len(patient_df)} {program_type} patients, transformed to {len(events_df)} events"
        )
    else:
        logging.warning(f"⚠️ No {program_type} patient data loaded")

    return result


def fetch_odk_data_for_user(
    user: dict, form_id: str = None
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Load ODK data from pre-fetched CSV files.
    """
    return {"odk_forms": {}}


def fetch_combined_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
    odk_form_id: str = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Load both program data and ODK data for a user.
    """
    result = {}

    if program_uid:
        program_data = fetch_program_data_for_user(
            user, program_uid, facility_uids, period_label
        )
        result.update(program_data)

    odk_data = fetch_odk_data_for_user(user, odk_form_id)
    result.update(odk_data)

    return result


def list_available_odk_forms() -> List[dict]:
    """List available ODK forms - empty for CSV-based system."""
    return []


def get_newborn_program_uid() -> str:
    """Get the newborn program UID from the database."""
    programs = get_all_programs()
    for program in programs:
        if program.get("program_name") == "Newborn Care Form":
            return program.get("program_uid", "")

    return "TIdYusMYiKl"


def get_patient_level_summary(patient_df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics from patient-level DataFrame."""
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
    }

    return summary


# Debug: List files in directories on startup
def list_available_files():
    """List available CSV files in each directory for debugging."""
    logging.info("📂 Scanning for available data files...")

    for program_type in ["maternal", "newborn"]:
        data_dir = get_data_directory(program_type)

        if os.path.exists(data_dir):
            files = glob.glob(os.path.join(data_dir, "*.csv"))

            if files:
                logging.info(
                    f"   📁 {program_type.upper()} files ({len(files)} found):"
                )
                for f in sorted(files):
                    file_size = os.path.getsize(f) / 1024
                    logging.info(f"      • {os.path.basename(f)} ({file_size:.1f} KB)")
            else:
                logging.warning(
                    f"   📁 {program_type.upper()} directory empty: {data_dir}"
                )
        else:
            logging.warning(
                f"   📁 {program_type.upper()} directory doesn't exist: {data_dir}"
            )


# List available files on startup
list_available_files()

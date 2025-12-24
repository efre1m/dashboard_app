# utils/data_service.py
from typing import Optional, Dict, List, Union, Set, Any
import pandas as pd
import logging
import os
import glob
import re

# ========== IMPROVED FACILITY NAME NORMALIZATION ==========


def normalize_facility_name(facility_name: str) -> str:
    """
    AGGRESSIVE normalization: Treat facilities as same if they mean the same thing
    regardless of spacing, capitalization, minor typos, or word order variations.

    Philosophy: "Abiadi General Hospital" == "abiadi general hospital" == "Abiadi   General   Hospital"
                == "General Hospital Abiadi" == "Abiadi GH"

    Returns a CONSISTENT normalized string for matching.
    """
    if not facility_name or pd.isna(facility_name):
        return ""

    # Convert to string and basic cleanup
    name = str(facility_name).strip()

    # 1. Convert to lowercase for case-insensitive comparison
    name = name.lower()

    # 2. Remove ALL special characters except letters, numbers, and spaces
    name = re.sub(r"[^\w\s]", "", name)

    # 3. Fix common typos and abbreviations
    typo_corrections = {
        "prrimary": "primary",
        "generl": "general",
        "hospial": "hospital",
        "hosp": "hospital",
        "referal": "referral",
        "specilized": "specialized",
        "medcal": "medical",
        "centr": "center",
        "clinc": "clinic",
        "despensary": "dispensary",
        "despensery": "dispensary",
        "univeristy": "university",
        "uni": "university",
        "govt": "government",
        "gov": "government",
        "moh": "ministry of health",
        "phc": "primary health center",
        "gh": "general hospital",
        "rh": "referral hospital",
        "sh": "specialized hospital",
        "ph": "primary hospital",
        "th": "teaching hospital",
        "uh": "university hospital",
        "mch": "maternal child health",
    }

    for typo, correction in typo_corrections.items():
        name = re.sub(r"\b" + typo + r"\b", correction, name)

    # 4. Standardize common facility type names (remove variations)
    facility_type_standardization = {
        "health center": "hc",
        "primary hospital": "ph",
        "general hospital": "gh",
        "referral hospital": "rh",
        "specialized hospital": "sh",
        "teaching hospital": "th",
        "university hospital": "uh",
        "medical center": "mc",
        "medical college": "mc",
        "clinic": "cl",
        "dispensary": "dp",
        "health post": "hp",
        "maternity": "mat",
        "child health": "ch",
        "maternal child health": "mch",
        "regional hospital": "rh",
        "zonal hospital": "zh",
        "district hospital": "dh",
        "health facility": "hf",
    }

    for old_type, new_type in facility_type_standardization.items():
        name = re.sub(r"\b" + old_type + r"\b", new_type, name)

    # 5. Remove common generic words that don't help identification
    generic_words = [
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "by",
        "for",
        "of",
        "to",
        "a",
        "an",
        "st",
        "saint",
        "dr",
        "doctor",
        "prof",
        "professor",
        "hospital",
        "center",
        "centre",
        "clinic",
        "facility",
        "health",
        "medical",
        "care",
        "unit",
        "department",
        "ward",
        "building",
    ]

    for word in generic_words:
        name = re.sub(r"\b" + word + r"\b", "", name)

    # 6. Extract only the UNIQUE IDENTIFYING part (facility name without type)
    # Keep only alphanumeric characters
    name = re.sub(r"[^a-z0-9]", "", name)

    # 7. Sort letters alphabetically to handle word order variations
    # This makes "Abiadi General Hospital" and "General Hospital Abiadi" match
    name = "".join(sorted(name))

    # 8. Return a consistent short code
    # If result is empty (shouldn't happen), return original hashed
    if not name:
        import hashlib

        return hashlib.md5(str(facility_name).encode()).hexdigest()[:8]

    return name


def normalize_all_facility_names(
    df: pd.DataFrame, facility_column: str = "orgUnit_name"
) -> pd.DataFrame:
    """Normalize facility names in dataframe with AGGRESSIVE matching"""
    if df.empty or facility_column not in df.columns:
        return df

    df = df.copy()

    print(f"\n🔧 NORMALIZING FACILITY NAMES (AGGRESSIVE MODE):")
    print(f"   Column: {facility_column}")
    print(f"   Unique names before: {df[facility_column].nunique()}")

    # Show examples of normalization
    sample_data = df[facility_column].dropna().unique()[:10]
    for original in sample_data:
        normalized = normalize_facility_name(original)
        print(f"     '{original}' → '{normalized}'")

    # Apply normalization
    df[facility_column + "_normalized"] = df[facility_column].apply(
        normalize_facility_name
    )

    print(
        f"   Unique normalized names: {df[facility_column + '_normalized'].nunique()}"
    )

    return df


def debug_facility_matching(user_facilities: List[str], csv_facilities: List[str]):
    """Debug function to show facility matching process"""
    print(f"\n{'='*100}")
    print(f"🔍 DEBUG: FACILITY MATCHING PROCESS")
    print(f"{'='*100}")

    print(f"\n📊 DATABASE FACILITIES ({len(user_facilities)}):")
    for i, fac in enumerate(user_facilities[:20]):
        normalized = normalize_facility_name(fac)
        print(f"   {i+1:2d}. '{fac}' → '{normalized}'")

    print(f"\n📊 CSV FACILITIES ({len(csv_facilities)}):")
    for i, fac in enumerate(csv_facilities[:20]):
        normalized = normalize_facility_name(fac)
        print(f"   {i+1:2d}. '{fac}' → '{normalized}'")

    # Find matches
    print(f"\n🔍 FINDING MATCHES:")

    # Create normalized sets
    db_normalized = {normalize_facility_name(f): f for f in user_facilities}
    csv_normalized = {normalize_facility_name(f): f for f in csv_facilities}

    matches = set(db_normalized.keys()) & set(csv_normalized.keys())
    db_only = set(db_normalized.keys()) - set(csv_normalized.keys())
    csv_only = set(csv_normalized.keys()) - set(db_normalized.keys())

    print(f"   ✅ MATCHES FOUND: {len(matches)}")
    for i, norm_key in enumerate(list(matches)[:10]):
        db_original = db_normalized.get(norm_key, "Unknown")
        csv_original = csv_normalized.get(norm_key, "Unknown")
        print(f"     {i+1}. DB: '{db_original}' ↔ CSV: '{csv_original}'")

    if db_only:
        print(f"\n   ❌ IN DB ONLY: {len(db_only)}")
        for i, norm_key in enumerate(list(db_only)[:10]):
            original = db_normalized.get(norm_key, "Unknown")
            print(f"     {i+1}. '{original}' → '{norm_key}'")

    if csv_only:
        print(f"\n   ❓ IN CSV ONLY: {len(csv_only)}")
        for i, norm_key in enumerate(list(csv_only)[:10]):
            original = csv_normalized.get(norm_key, "Unknown")
            print(f"     {i+1}. '{original}' → '{norm_key}'")

    print(f"\n{'='*100}")
    print(f"🔍 DEBUG: END FACILITY MATCHING")
    print(f"{'='*100}\n")


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


# Replace ONLY the filter_patient_data_by_user function in your current data_service.py
def filter_patient_data_by_user(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    ✅ FIXED: Filter by orgUnit UID instead of facility names.
    Uses 'orgUnit' column which has exact DHIS2 UIDs.
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

        # Get user facilities with UIDs
        from utils.queries import get_facilities_for_user

        user_facilities_raw = get_facilities_for_user(user)

        # Extract UIDs
        user_facility_uids = []
        for facility in user_facilities_raw:
            if isinstance(facility, (list, tuple)) and len(facility) >= 2:
                user_facility_uids.append(facility[1])  # UID is second element

        logging.info(
            f"🔍 Looking for facilities in region '{region_name}': {len(user_facility_uids)} facilities"
        )

        if "orgUnit" in df.columns and user_facility_uids:
            # Filter by UID
            filtered_df = df[df["orgUnit"].isin(user_facility_uids)]

            logging.info(
                f"🏞️ Regional user '{region_name}' - filtered to {len(filtered_df)} patients"
            )
            return filtered_df
        else:
            if "orgUnit" not in df.columns:
                logging.warning(
                    "DataFrame has no orgUnit column - cannot filter by UID"
                )
            return pd.DataFrame()

    elif user_role == "facility":
        # For facility users, we need to find their facility UID
        from utils.queries import get_facilities_for_user

        user_facilities_raw = get_facilities_for_user(user)

        user_facility_uid = None
        if (
            user_facilities_raw
            and isinstance(user_facilities_raw[0], (list, tuple))
            and len(user_facilities_raw[0]) >= 2
        ):
            user_facility_uid = user_facilities_raw[0][1]  # UID is second element

        if not user_facility_uid:
            logging.warning("Facility user has no facility UID - cannot filter data")
            return pd.DataFrame()

        if "orgUnit" in df.columns:
            # Filter by single UID
            filtered_df = df[df["orgUnit"] == user_facility_uid]

            if not filtered_df.empty:
                logging.info(f"🏥 Facility user - found {len(filtered_df)} patients")
                return filtered_df
            else:
                logging.warning(
                    f"⚠️ No data found for facility UID: {user_facility_uid}"
                )
                return pd.DataFrame()
        else:
            logging.warning("DataFrame has no orgUnit column - cannot filter")
            return pd.DataFrame()

    else:
        logging.warning(f"Unknown user role '{user_role}' - returning no data")
        return pd.DataFrame()


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

    ✅ FIXED: No transformation needed - we work directly with patient data
    """
    from utils.queries import get_program_by_uid

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

    # ✅ NO TRANSFORMATION NEEDED - we work directly with patient data
    # Create empty events DataFrame for compatibility
    events_df = pd.DataFrame()

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
        "events": events_df,  # Empty - we don't need transformed events
        "patients": patient_df,  # Original patient-level data
        "optimization_stats": {
            "data_source": "pre_fetched_csv",
            "program_type": program_type,
            "patient_count": len(patient_df),
            "events_count": 0,  # We don't have events
        },
    }

    if not patient_df.empty:
        logging.info(
            f"✅ Successfully loaded {len(patient_df)} {program_type} patients"
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
    from utils.queries import get_all_programs

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

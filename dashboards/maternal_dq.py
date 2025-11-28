# dashboards/maternal_dq.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
from utils.patient_mapping import get_patient_name_from_tei

# Focus only on numerical data elements from fetched data
MATERNAL_NUMERICAL_ELEMENTS = {
    "QUlJEvzGcQK": {
        "name": "Birth Weight",
        "impossible_range": (500, 8000),
        "unit": "grams",
    },
}

# Required maternal data elements from fetched data - Only check First Reason
MATERNAL_REQUIRED_ELEMENTS = {
    "Q1p7CxWGUoi": "FP Counseling and Method Provided pp",
    "lphtwP2ViZU": "Mode of Delivery maternal",
    "wZig9cek3Gv": "Birth Outcome",
    "VzwnSBROvUm": "Number of Newborns",
    "z7Eb2yFLOBI": "Date/stay pp",
    "TjQOcW6tm8k": "Condition of Discharge",
    "CJiTafFo0TS": "Obstetric condition at delivery",
    "yVRLuRU943e": "Uterotonics given",
    "tTrH9cOQRnZ": "HIV Result",
    "H7J2SxBpObS": "ARV Rx for Newborn (By type) pp",
    "QUlJEvzGcQK": "Birth Weight (grams)",
}

# Cache TTL for data quality results (1 hour)
DQ_CACHE_TTL = 3600


def get_user_cache_key(user, analysis_type):
    """Generate cache key specific to user and analysis type"""
    user_identifier = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    return f"dq_{analysis_type}_{user_identifier}"


def clear_dq_cache(user=None):
    """Clear DQ cache for specific user or all users"""
    if user:
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith(f"dq_") and key.endswith(user_key)
        ]
        for key in keys_to_clear:
            del st.session_state[key]
        logging.info(f"🧹 Cleared DQ cache for user: {user_key}")
    else:
        # Clear all DQ caches
        keys_to_clear = [
            key for key in st.session_state.keys() if key.startswith("dq_")
        ]
        for key in keys_to_clear:
            del st.session_state[key]
        logging.info("🧹 Cleared ALL DQ caches")


def safe_numeric_conversion(value):
    """Safely convert value to numeric"""
    if pd.isna(value) or value is None or value == "":
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    try:
        cleaned_value = str(value).strip()
        return float(cleaned_value)
    except (ValueError, TypeError):
        return np.nan


def get_region_from_facility(facility_name):
    """Get region name from facility name - optimized for both regional and national users"""
    if (
        hasattr(st, "session_state")
        and "facility_to_region_map" in st.session_state
        and facility_name in st.session_state.facility_to_region_map
    ):
        return st.session_state.facility_to_region_map[facility_name]
    return "Unknown Region"


def is_national_user(user):
    """Check if user has national-level access"""
    role = user.get("role", "").lower()
    country_name = user.get("country_name", "")
    region_name = user.get("region_name", "")

    # User is national if they have country_name but no specific region
    # or if their role indicates national level access
    national_roles = ["national", "admin", "supervisor", "coordinator"]
    return (country_name and not region_name) or any(
        national_role in role for national_role in national_roles
    )


def check_maternal_outliers(events_df, user):
    """Check for outliers in maternal numerical data elements with caching - SUPPORTS NATIONAL USERS"""
    if events_df.empty:
        return pd.DataFrame()

    cache_key = get_user_cache_key(user, "maternal_outliers")

    # Check cache
    if (
        cache_key in st.session_state
        and time.time() - st.session_state[f"{cache_key}_timestamp"] < DQ_CACHE_TTL
    ):
        logging.info("✅ Using cached maternal outliers")
        return st.session_state[cache_key].copy()

    outliers = []

    # ✅ FIX: National users see all regions, regional users see only their region
    user_region = user.get("region_name", "Unknown Region")
    is_national = is_national_user(user)

    logging.info(
        f"🔍 Maternal DQ - User region: {user_region}, Is national: {is_national}"
    )

    for data_element_uid, element_info in MATERNAL_NUMERICAL_ELEMENTS.items():
        element_name = element_info["name"]
        min_val, max_val = element_info["impossible_range"]
        unit = element_info["unit"]

        element_rows = events_df[
            events_df["dataElement_uid"] == data_element_uid
        ].copy()
        if not element_rows.empty:
            element_rows["numeric_value"] = element_rows["value"].apply(
                safe_numeric_conversion
            )
            numeric_rows = element_rows[element_rows["numeric_value"].notna()]

            if not numeric_rows.empty:
                outlier_mask = (numeric_rows["numeric_value"] < min_val) | (
                    numeric_rows["numeric_value"] > max_val
                )
                outlier_rows = numeric_rows[outlier_mask]

                for _, row in outlier_rows.iterrows():
                    outlier_value = row["numeric_value"]
                    issue_type = (
                        f"Too Low (< {min_val} {unit})"
                        if outlier_value < min_val
                        else f"Too High (> {max_val} {unit})"
                    )
                    tei_id = row.get("tei_id")
                    first_name, father_name = get_patient_name_from_tei(
                        tei_id, "maternal"
                    )
                    facility_name = row.get("orgUnit_name", "Unknown Facility")
                    region = get_region_from_facility(facility_name)

                    # ✅ FIX: National users see all regions, regional users see only their region
                    if is_national or region == user_region:
                        outliers.append(
                            {
                                "First Name": first_name,
                                "Father Name": father_name,
                                "Region": region,
                                "Facility": facility_name,
                                "Data Element": element_name,
                                "Value": outlier_value,
                                "Unit": unit,
                                "Issue Type": issue_type,
                                "TEI ID": tei_id,
                            }
                        )

    result_df = pd.DataFrame(outliers)

    # Cache the results
    st.session_state[cache_key] = result_df.copy()
    st.session_state[f"{cache_key}_timestamp"] = time.time()

    logging.info(
        f"✅ Maternal outliers - National user: {is_national}, Found: {len(result_df)} outliers"
    )
    return result_df


def check_missing_data_elements(events_df, user):
    """OPTIMIZED CHECK: Batch process missing data elements for better performance"""
    if events_df.empty:
        return pd.DataFrame()

    cache_key = get_user_cache_key(user, "maternal_missing_elements")

    # Check cache
    if (
        cache_key in st.session_state
        and time.time() - st.session_state[f"{cache_key}_timestamp"] < DQ_CACHE_TTL
    ):
        logging.info("✅ Using cached maternal missing elements")
        return st.session_state[cache_key].copy()

    logging.info("🔍 Starting OPTIMIZED maternal missing data check")

    # ✅ OPTIMIZATION 1: Pre-filter for required data elements only
    required_uids = list(MATERNAL_REQUIRED_ELEMENTS.keys())
    filtered_events = events_df[events_df["dataElement_uid"].isin(required_uids)].copy()

    if filtered_events.empty:
        logging.info("✅ No required data elements found in events")
        return pd.DataFrame()

    # ✅ OPTIMIZATION 2: Filter out non-empty values in one go
    missing_mask = filtered_events["value"].isna() | (filtered_events["value"] == "")
    missing_rows = filtered_events[missing_mask].copy()

    if missing_rows.empty:
        logging.info("✅ No missing data elements found")
        return pd.DataFrame()

    # ✅ OPTIMIZATION 3: Batch process patient names
    unique_tei_ids = missing_rows["tei_id"].unique()
    patient_names = {}

    # Batch get patient names for all unique TEI IDs
    for tei_id in unique_tei_ids:
        first_name, father_name = get_patient_name_from_tei(tei_id, "maternal")
        patient_names[tei_id] = (first_name, father_name)

    # ✅ OPTIMIZATION 4: Pre-compute regions for facilities
    unique_facilities = missing_rows["orgUnit_name"].unique()
    facility_regions = {}

    for facility in unique_facilities:
        facility_regions[facility] = get_region_from_facility(facility)

    # ✅ OPTIMIZATION 5: User region filtering
    user_region = user.get("region_name", "Unknown Region")
    is_national = is_national_user(user)

    missing_data = []

    # ✅ OPTIMIZATION 6: Vectorized processing with pre-computed values
    for _, row in missing_rows.iterrows():
        tei_id = row.get("tei_id")
        facility_name = row.get("orgUnit_name", "Unknown Facility")
        region = facility_regions.get(facility_name, "Unknown Region")

        # Apply user region filter
        if not is_national and region != user_region:
            continue

        data_element_uid = row.get("dataElement_uid")
        element_name = MATERNAL_REQUIRED_ELEMENTS.get(
            data_element_uid, "Unknown Element"
        )

        first_name, father_name = patient_names.get(tei_id, ("Unknown", "Unknown"))

        missing_data.append(
            {
                "First Name": first_name,
                "Father Name": father_name,
                "Region": region,
                "Facility": facility_name,
                "TEI ID": tei_id,
                "Issue Type": "Missing Data Element",
                "Data Element Missing": element_name,
            }
        )

    result_df = pd.DataFrame(missing_data)

    # Cache the results
    st.session_state[cache_key] = result_df.copy()
    st.session_state[f"{cache_key}_timestamp"] = time.time()

    logging.info(
        f"✅ OPTIMIZED Maternal - Found {len(result_df)} missing data elements"
    )
    return result_df


def render_maternal_data_quality():
    """Render Maternal Data Quality Analysis"""
    if (
        "maternal_events_df" not in st.session_state
        or st.session_state.maternal_events_df.empty
        or "maternal_tei_df" not in st.session_state
        or st.session_state.maternal_tei_df.empty
    ):
        st.warning("No maternal data available")
        return

    user = st.session_state.get("user", {})

    # ✅ ADD: Show user level info
    is_national = is_national_user(user)
    user_level = (
        "National Level"
        if is_national
        else f"Regional Level ({user.get('region_name', 'Unknown Region')})"
    )
    st.sidebar.info(f"👤 Data Quality: {user_level}")

    tab1, tab2 = st.tabs(["Outliers", "Missing Data"])

    with tab1:
        render_outliers_tab(user)
    with tab2:
        render_missing_elements_tab(user)


def render_outliers_tab(user):
    """Render the outliers analysis tab"""
    events_df = st.session_state.maternal_events_df
    outliers_df = check_maternal_outliers(events_df, user)

    if outliers_df.empty:
        st.success("No data outliers found")
        return

    st.error(f"Found {len(outliers_df)} data outliers")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        all_regions = ["All Regions"] + sorted(outliers_df["Region"].unique())
        selected_region = st.selectbox(
            "Region", options=all_regions, index=0, key="maternal_outlier_region"
        )
    with col2:
        all_facilities = ["All Facilities"] + sorted(outliers_df["Facility"].unique())
        selected_facility = st.selectbox(
            "Facility", options=all_facilities, index=0, key="maternal_outlier_facility"
        )
    with col3:
        all_elements = ["All Data Elements"] + sorted(
            outliers_df["Data Element"].unique()
        )
        selected_element = st.selectbox(
            "Data Element",
            options=all_elements,
            index=0,
            key="maternal_outlier_element",
        )

    # Apply filters
    filtered_df = outliers_df.copy()
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_facility != "All Facilities":
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]
    if selected_element != "All Data Elements":
        filtered_df = filtered_df[filtered_df["Data Element"] == selected_element]

    if filtered_df.empty:
        st.info("No data outliers match filters")
        return

    # Display table - REMOVED the additional "No" column
    display_df = filtered_df.copy().reset_index(drop=True)
    display_columns = [
        "First Name",
        "Father Name",
        "Region",
        "Facility",
        "Data Element",
        "Value",
        "Unit",
        "Issue Type",
    ]

    st.dataframe(display_df[display_columns], use_container_width=True)


def render_missing_elements_tab(user):
    """Render the missing data elements analysis tab"""
    events_df = st.session_state.maternal_events_df
    missing_elements_df = check_missing_data_elements(events_df, user)

    if missing_elements_df.empty:
        st.success("All data elements complete")
        return

    st.error(f"Found {len(missing_elements_df)} missing data elements")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_regions = ["All Regions"] + sorted(missing_elements_df["Region"].unique())
        selected_region = st.selectbox(
            "Region",
            options=all_regions,
            index=0,
            key="maternal_missing_elements_region",
        )
    with col2:
        all_facilities = ["All Facilities"] + sorted(
            missing_elements_df["Facility"].unique()
        )
        selected_facility = st.selectbox(
            "Facility",
            options=all_facilities,
            index=0,
            key="maternal_missing_elements_facility",
        )

    # Apply filters
    filtered_df = missing_elements_df.copy()
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_facility != "All Facilities":
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]

    if filtered_df.empty:
        st.info("No missing elements match filters")
        return

    # Display table - REMOVED the additional "No" column
    display_df = filtered_df.copy().reset_index(drop=True)
    display_columns = [
        "First Name",
        "Father Name",
        "Region",
        "Facility",
        "Data Element Missing",
    ]

    st.dataframe(display_df[display_columns], use_container_width=True)

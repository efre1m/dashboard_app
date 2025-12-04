# dashboards/newborn_dq.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
from utils.patient_mapping import get_patient_name_from_tei

# Focus only on numerical data elements from fetched data
NEWBORN_NUMERICAL_ELEMENTS = {
    "yxWUMt3sCil": {
        "name": "Weight on Admission",
        "impossible_range": (400, 6000),
        "unit": "grams",
    },
    "gZi9y12E9i7": {
        "name": "Temperature on Admission",
        "impossible_range": (28, 42),
        "unit": "°C",
    },
    "yBCwmQP0A6a": {
        "name": "Discharge Weight",
        "impossible_range": (400, 6000),
        "unit": "grams",
    },
    "nIKIu6f5vbW": {
        "name": "Lowest recorded temperature",
        "impossible_range": (28, 42),
        "unit": "°C",
    },
}

# Required newborn data elements from fetched data - with program stages
NEWBORN_REQUIRED_ELEMENTS = {
    # Admission Information (l39SlVGlQGs)
    "UOmhJkyAK6h": "Date of Admission",
    "yxWUMt3sCil": "Weight on admission",
    "T30GbTiVgFR": "First Reason for Admission",
    "OpHw2X58x5i": "Second Reason for Admission",
    "gJH6PkYI6IV": "Third Reason for Admission",
    "aK5txmRYpVX": "Birth location (inborn/outborn)",
    # Observations And Nursing Care 1 (j0HI2eJjvbj)
    "gZi9y12E9i7": "Temperature on admission",
    # Interventions (ed8ErpgTCwx)
    "QK7Fi6OwtDC": "KMC Administered",
    "wlHEf9FdmJM": "CPAP Administered",
    "sxtsEDilKZd": "Were antibiotics administered?",
    # Discharge And Final Diagnosis (TOicTEwzSGj)
    "vmOAGuFcaz4": "Newborn status at discharge",
    "yBCwmQP0A6a": "Discharge Weight",
    "wn0tHaHcceW": "Sub-Categories of Infection",
    # Observations And Nursing Care 2 (VsVlpG1V2ub)
    "nIKIu6f5vbW": "Lowest recorded temperature",
    # Microbiology And Labs (aCrttmnx7FI)
    "A94ibeuO9GL": "Blood culture for suspected sepsis",
}

# Mapping of data elements to program stage names (from data_service.py)
NEWBORN_ELEMENT_TO_PROGRAM_STAGE = {
    # Admission Information
    "UOmhJkyAK6h": "Admission Information",
    "yxWUMt3sCil": "Admission Information",
    "T30GbTiVgFR": "Admission Information",
    "OpHw2X58x5i": "Admission Information",
    "gJH6PkYI6IV": "Admission Information",
    "aK5txmRYpVX": "Admission Information",
    # Observations And Nursing Care 1
    "gZi9y12E9i7": "Observations And Nursing Care 1",
    # Interventions
    "QK7Fi6OwtDC": "Interventions",
    "wlHEf9FdmJM": "Interventions",
    "sxtsEDilKZd": "Interventions",
    # Discharge And Final Diagnosis
    "vmOAGuFcaz4": "Discharge And Final Diagnosis",
    "yBCwmQP0A6a": "Discharge And Final Diagnosis",
    "wn0tHaHcceW": "Discharge And Final Diagnosis",
    # Observations And Nursing Care 2
    "nIKIu6f5vbW": "Observations And Nursing Care 2",
    # Microbiology And Labs
    "A94ibeuO9GL": "Microbiology And Labs",
}

# Group admission reason elements to check if ANY is present
ADMISSION_REASON_ELEMENTS = ["T30GbTiVgFR", "OpHw2X58x5i", "gJH6PkYI6IV"]

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


def get_program_stage_for_element(data_element_uid):
    """Get program stage name for a data element"""
    return NEWBORN_ELEMENT_TO_PROGRAM_STAGE.get(
        data_element_uid, "Unknown Program Stage"
    )


def check_newborn_outliers(events_df, user):
    """Check for outliers in newborn numerical data elements - Only show TEIs with outliers"""
    if events_df.empty:
        return pd.DataFrame()

    cache_key = get_user_cache_key(user, "newborn_outliers")

    # Check cache
    if (
        cache_key in st.session_state
        and time.time() - st.session_state[f"{cache_key}_timestamp"] < DQ_CACHE_TTL
    ):
        logging.info("✅ Using cached newborn outliers")
        return st.session_state[cache_key].copy()

    # User region filtering setup
    user_region = user.get("region_name", "Unknown Region")
    is_national = is_national_user(user)

    logging.info(
        f"🔍 Newborn DQ - User region: {user_region}, Is national: {is_national}"
    )

    # Get all unique TEIs in the dataset
    all_teis = events_df["tei_id"].unique()

    # FOR NATIONAL USERS: Sample 2000 TEIs for outliers (increased for better detection)
    outlier_sample_size = 2000  # Increased to 2000 for outliers

    if is_national and len(all_teis) > outlier_sample_size:
        logging.info(
            f"🌍 National user: Sampling {outlier_sample_size} TEIs from {len(all_teis)} total for outliers"
        )
        selected_teis = np.random.choice(
            all_teis, size=outlier_sample_size, replace=False
        )
        all_teis = selected_teis
        logging.info(
            f"📊 Sampling completed: {len(all_teis)} TEIs to check for outliers"
        )

    # Create a dictionary to track data for each TEI
    tei_data = {}
    # Track all possible outlier columns across all TEIs
    all_outlier_columns = set()

    outliers_found = 0
    teis_processed = 0

    # Process each TEI
    for tei_id in all_teis:
        teis_processed += 1

        # Get patient name
        first_name, last_name = get_patient_name_from_tei(tei_id, "newborn")

        # Get facility and region for this TEI
        tei_events = events_df[events_df["tei_id"] == tei_id]
        if not tei_events.empty:
            facility_name = tei_events.iloc[0].get("orgUnit_name", "Unknown Facility")
            region = get_region_from_facility(facility_name)

            # Apply user region filter early (for regional users)
            if not is_national and region != user_region:
                continue

            # Track outlier count for this TEI
            outlier_count = 0
            tei_outlier_data = {}

            # Check each numerical data element for outliers
            for data_element_uid, element_info in NEWBORN_NUMERICAL_ELEMENTS.items():
                element_name = element_info["name"]
                min_val, max_val = element_info["impossible_range"]
                unit = element_info.get("unit", "")
                program_stage = get_program_stage_for_element(data_element_uid)

                # Create column name in format: ElementName_ProgramStage_Unit
                if unit:
                    column_name = f"{element_name}_{program_stage}_{unit}"
                else:
                    column_name = f"{element_name}_{program_stage}"

                # Check if element exists for this TEI
                element_rows = tei_events[
                    tei_events["dataElement_uid"] == data_element_uid
                ]

                if not element_rows.empty:
                    # Convert value to numeric
                    for _, row in element_rows.iterrows():
                        value = row.get("value")
                        numeric_value = safe_numeric_conversion(value)

                        if pd.notna(numeric_value):
                            # Check if it's an outlier
                            is_outlier = False
                            issue_type = ""

                            if numeric_value < min_val:
                                is_outlier = True
                                issue_type = f"Too Low (< {min_val} {unit})"
                            elif numeric_value > max_val:
                                is_outlier = True
                                issue_type = f"Too High (> {max_val} {unit})"

                            # If it's an outlier, add to the TEI data
                            if is_outlier:
                                # Format: value - issue type
                                formatted_value = f"{numeric_value:.2f} - {issue_type}"
                                tei_outlier_data[column_name] = formatted_value
                                all_outlier_columns.add(column_name)
                                outlier_count += 1
                                outliers_found += 1

            # ONLY add TEI to results if it has at least one outlier
            if outlier_count > 0:
                # Initialize TEI data structure with basic info
                tei_data[tei_id] = {
                    "First Name": first_name,
                    "Last Name": last_name,
                    "Region": region,
                    "Facility": facility_name,
                    "TEI ID": tei_id,
                    "Outlier Count": outlier_count,
                    **tei_outlier_data,  # Add all outlier columns
                }

    # Convert to DataFrame
    if not tei_data:
        result_df = pd.DataFrame()
    else:
        result_df = pd.DataFrame(list(tei_data.values()))

        # Ensure all outlier columns exist in DataFrame (fill with empty for TEIs that don't have that specific outlier)
        for column in all_outlier_columns:
            if column not in result_df.columns:
                result_df[column] = ""

        # Sort by outlier count (highest first)
        result_df = result_df.sort_values("Outlier Count", ascending=False)

    # Cache the results
    st.session_state[cache_key] = result_df.copy()
    st.session_state[f"{cache_key}_timestamp"] = time.time()

    logging.info(
        f"✅ Newborn outliers - Processed {teis_processed} TEIs, found {len(result_df)} TEIs with {outliers_found} total outliers"
    )

    # Add info message for national users
    if is_national:
        if result_df.empty:
            logging.info(
                f"ℹ️ National user: No outliers found in sample of {len(all_teis)} patients"
            )
        else:
            logging.info(
                f"🌍 National user: Found {len(result_df)} patients with outliers in sample of {len(all_teis)}"
            )

    return result_df


def check_missing_data_elements(events_df, user):
    """Create matrix view with one row per TEI - Only show TEIs with missing data"""
    if events_df.empty:
        return pd.DataFrame()

    cache_key = get_user_cache_key(user, "newborn_missing_elements_matrix")

    # Check cache
    if (
        cache_key in st.session_state
        and time.time() - st.session_state[f"{cache_key}_timestamp"] < DQ_CACHE_TTL
    ):
        logging.info("✅ Using cached newborn missing elements matrix")
        return st.session_state[cache_key].copy()

    logging.info("🔍 Creating matrix view for newborn missing data")

    # Get all unique TEIs in the dataset
    all_teis = events_df["tei_id"].unique()

    # FOR NATIONAL USERS: Sample 500 TEIs (increased from 100)
    is_national = is_national_user(user)
    sample_size = 500

    if is_national and len(all_teis) > sample_size:
        logging.info(
            f"🌍 National user: Sampling {sample_size} TEIs from {len(all_teis)} total for missing data"
        )
        selected_teis = np.random.choice(all_teis, size=sample_size, replace=False)
        all_teis = selected_teis

    # User region filtering setup
    user_region = user.get("region_name", "Unknown Region")

    # Create a dictionary to track data for each TEI
    tei_data = {}
    # Track all possible missing element columns across all TEIs
    all_missing_columns = set()

    # Process each TEI
    for tei_id in all_teis:
        # Get patient name
        first_name, last_name = get_patient_name_from_tei(tei_id, "newborn")

        # Get facility and region for this TEI
        tei_events = events_df[events_df["tei_id"] == tei_id]
        if not tei_events.empty:
            facility_name = tei_events.iloc[0].get("orgUnit_name", "Unknown Facility")
            region = get_region_from_facility(facility_name)

            # Apply user region filter early (for regional users)
            if not is_national and region != user_region:
                continue

            # Track missing elements for this TEI
            missing_elements = []

            # Check each required data element
            for data_element_uid, element_name in NEWBORN_REQUIRED_ELEMENTS.items():
                program_stage = get_program_stage_for_element(data_element_uid)
                column_name = f"{element_name}_{program_stage}"

                # Check if element exists for this TEI
                element_rows = tei_events[
                    tei_events["dataElement_uid"] == data_element_uid
                ]

                is_missing = False
                if element_rows.empty:
                    # Element not found at all
                    is_missing = True
                else:
                    # Check if value is present
                    has_value = False
                    for _, row in element_rows.iterrows():
                        value = row.get("value")
                        if pd.notna(value) and str(value).strip() != "":
                            has_value = True
                            break

                    if not has_value:
                        is_missing = True

                # Track missing elements
                if is_missing:
                    missing_elements.append(column_name)

            # ONLY add TEI to results if it has at least one missing element
            if missing_elements:
                # Initialize TEI data structure with basic info
                tei_data[tei_id] = {
                    "First Name": first_name,
                    "Last Name": last_name,
                    "Region": region,
                    "Facility": facility_name,
                    "TEI ID": tei_id,
                    "Missing Count": len(missing_elements),
                }

                # Add all missing element columns
                for column_name in missing_elements:
                    tei_data[tei_id][column_name] = "missing"
                    all_missing_columns.add(column_name)

    # Convert to DataFrame
    if not tei_data:
        result_df = pd.DataFrame()
    else:
        result_df = pd.DataFrame(list(tei_data.values()))

        # Ensure all missing columns exist in DataFrame (fill with empty for TEIs that don't have that missing element)
        for column in all_missing_columns:
            if column not in result_df.columns:
                result_df[column] = ""

        # Sort by missing count (highest first)
        result_df = result_df.sort_values("Missing Count", ascending=False)

    # Cache the results
    st.session_state[cache_key] = result_df.copy()
    st.session_state[f"{cache_key}_timestamp"] = time.time()

    logging.info(
        f"✅ Newborn missing data - {len(result_df)} TEIs with missing data (sampled {len(all_teis)} TEIs)"
    )

    # Add info message for national users
    if is_national:
        if result_df.empty:
            logging.info(
                "ℹ️ National user: No missing data found in sample of 500 patients"
            )
        else:
            logging.info(
                f"🌍 National user: Found {len(result_df)} patients with missing data in sample of 500"
            )

    return result_df


def render_newborn_data_quality():
    """Render Newborn Data Quality Analysis"""
    if (
        "newborn_events_df" not in st.session_state
        or st.session_state.newborn_events_df.empty
        or "newborn_tei_df" not in st.session_state
        or st.session_state.newborn_tei_df.empty
    ):
        st.warning("No newborn data available")
        return

    user = st.session_state.get("user", {})

    # Show user level info
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
    events_df = st.session_state.newborn_events_df
    outliers_df = check_newborn_outliers(events_df, user)

    if outliers_df.empty:
        st.success("No data outliers found")
        return

    st.error(f"Found {len(outliers_df)} patients with data outliers")

    # Calculate summary statistics
    if "Outlier Count" in outliers_df.columns:
        total_outliers = outliers_df["Outlier Count"].sum()
        avg_outliers = outliers_df["Outlier Count"].mean()
        st.info(
            f"**Summary**: {total_outliers} total outlier values across {len(outliers_df)} patients (average: {avg_outliers:.1f} per patient)"
        )

    # Identify outlier columns (all columns that are not basic info)
    basic_columns = [
        "First Name",
        "Last Name",
        "Region",
        "Facility",
        "TEI ID",
        "Outlier Count",
    ]
    outlier_columns = [col for col in outliers_df.columns if col not in basic_columns]

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        if "Region" in outliers_df.columns:
            all_regions = ["All Regions"] + sorted(outliers_df["Region"].unique())
            selected_region = st.selectbox(
                "Region", options=all_regions, index=0, key="newborn_outlier_region"
            )
        else:
            selected_region = "All Regions"

    with col2:
        if "Facility" in outliers_df.columns:
            all_facilities = ["All Facilities"] + sorted(
                outliers_df["Facility"].unique()
            )
            selected_facility = st.selectbox(
                "Facility",
                options=all_facilities,
                index=0,
                key="newborn_outlier_facility",
            )
        else:
            selected_facility = "All Facilities"

    with col3:
        if "Outlier Count" in outliers_df.columns:
            outlier_counts = sorted(outliers_df["Outlier Count"].unique())
            all_counts = ["All Counts"] + [str(c) for c in outlier_counts]
            selected_count = st.selectbox(
                "Outlier Count",
                options=all_counts,
                index=0,
                key="newborn_outlier_count",
            )
        else:
            selected_count = "All Counts"

    # Apply filters
    filtered_df = outliers_df.copy()
    if selected_region != "All Regions" and "Region" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]
    if selected_count != "All Counts":
        filtered_df = filtered_df[filtered_df["Outlier Count"] == int(selected_count)]

    if filtered_df.empty:
        st.info("No data outliers match filters")
        return

    # Reorder columns: basic info first, then outlier elements
    display_columns = basic_columns + sorted(outlier_columns)
    display_columns = [col for col in display_columns if col in filtered_df.columns]

    # Create display dataframe
    display_df = filtered_df[display_columns].copy()

    # Color coding function for outlier values
    def style_outlier_cells(val):
        if (
            isinstance(val, str) and "- Too" in val
        ):  # Matches "2.0 - Too Low" or "2.0 - Too High"
            return "background-color: #ffcccc; color: #990000; font-weight: bold"
        return ""

    # Display with styling
    st.dataframe(
        display_df.style.map(style_outlier_cells, subset=outlier_columns),
        use_container_width=True,
        height=400,
    )


def render_missing_elements_tab(user):
    """Render the missing data elements analysis tab with matrix view"""
    events_df = st.session_state.newborn_events_df
    missing_elements_df = check_missing_data_elements(events_df, user)

    if missing_elements_df.empty:
        st.success("All data elements complete")
        return

    st.error(f"Found {len(missing_elements_df)} patients with missing data elements")

    # Calculate summary statistics
    if "Missing Count" in missing_elements_df.columns:
        total_missing = missing_elements_df["Missing Count"].sum()
        avg_missing = missing_elements_df["Missing Count"].mean()
        st.info(
            f"**Summary**: {total_missing} total missing elements across {len(missing_elements_df)} patients (average: {avg_missing:.1f} per patient)"
        )

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        if "Region" in missing_elements_df.columns:
            all_regions = ["All Regions"] + sorted(
                missing_elements_df["Region"].unique()
            )
            selected_region = st.selectbox(
                "Region", options=all_regions, index=0, key="newborn_missing_region"
            )
        else:
            selected_region = "All Regions"

    with col2:
        if "Facility" in missing_elements_df.columns:
            all_facilities = ["All Facilities"] + sorted(
                missing_elements_df["Facility"].unique()
            )
            selected_facility = st.selectbox(
                "Facility",
                options=all_facilities,
                index=0,
                key="newborn_missing_facility",
            )
        else:
            selected_facility = "All Facilities"

    with col3:
        if "Missing Count" in missing_elements_df.columns:
            missing_counts = sorted(missing_elements_df["Missing Count"].unique())
            all_counts = ["All Counts"] + [str(c) for c in missing_counts]
            selected_count = st.selectbox(
                "Missing Elements Count",
                options=all_counts,
                index=0,
                key="newborn_missing_count",
            )
        else:
            selected_count = "All Counts"

    # Apply filters
    filtered_df = missing_elements_df.copy()
    if selected_region != "All Regions" and "Region" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]
    if selected_count != "All Counts":
        filtered_df = filtered_df[filtered_df["Missing Count"] == int(selected_count)]

    if filtered_df.empty:
        st.info("No missing elements match the selected filters")
        return

    # Identify missing element columns
    basic_columns = [
        "First Name",
        "Last Name",
        "Region",
        "Facility",
        "TEI ID",
        "Missing Count",
    ]
    missing_element_columns = [
        col
        for col in filtered_df.columns
        if col not in basic_columns and (filtered_df[col] == "missing").any()
    ]

    # Reorder columns: basic info first, then missing elements
    display_columns = basic_columns + sorted(missing_element_columns)
    display_columns = [col for col in display_columns if col in filtered_df.columns]

    # Create display dataframe
    display_df = filtered_df[display_columns].copy()

    # Color coding function
    def style_missing_cells(val):
        if val == "missing":
            return "background-color: #ffcccc; color: #990000; font-weight: bold"
        return ""

    # Display with styling
    st.dataframe(
        display_df.style.map(style_missing_cells, subset=missing_element_columns),
        use_container_width=True,
        height=400,
    )

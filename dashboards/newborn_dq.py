# dashboards/newborn_dq.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
from utils.patient_mapping import get_patient_name_from_tei

# Focus only on numerical data elements from fetched data
NEWBORN_NUMERICAL_ELEMENTS = {
    "yxWUMt3sCil": {
        "name": "Weight on Admission",
        "impossible_range": (400, 6000),
        "unit": "grams",
    },
    "QUlJEvzGcQK": {
        "name": "Birth Weight",
        "impossible_range": (400, 8000),
        "unit": "grams",
    },
    "gZi9y12E9i7": {
        "name": "Temperature on Admission",
        "impossible_range": (28, 42),
        "unit": "°C",
    },
}

# Required newborn data elements from fetched data
NEWBORN_REQUIRED_ELEMENTS = {
    "QK7Fi6OwtDC": "KMC Administered",
    "yxWUMt3sCil": "Weight on admission",
    "gZi9y12E9i7": "Temperature on admission (°C)",
    "UOmhJkyAK6h": "Date of Admission",
    "wlHEf9FdmJM": "CPAP Administered",
    "T30GbTiVgFR": "First Reason for Admission",
    "OpHw2X58x5i": "Second Reason for Admission",
    "gJH6PkYI6IV": "Third Reason for Admission",
}


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
    """Get region name from facility name"""
    if (
        hasattr(st, "session_state")
        and "facility_to_region_map" in st.session_state
        and facility_name in st.session_state.facility_to_region_map
    ):
        return st.session_state.facility_to_region_map[facility_name]
    return "Unknown Region"


def check_newborn_outliers(events_df):
    """Check for outliers in newborn numerical data elements"""
    if events_df.empty:
        return pd.DataFrame()

    outliers = []
    for data_element_uid, element_info in NEWBORN_NUMERICAL_ELEMENTS.items():
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
                    first_name, last_name = get_patient_name_from_tei(tei_id, "newborn")
                    facility_name = row.get("orgUnit_name", "Unknown Facility")
                    region = get_region_from_facility(facility_name)

                    outliers.append(
                        {
                            "First Name": first_name,
                            "Last Name": last_name,
                            "Region": region,
                            "Facility": facility_name,
                            "Data Element": element_name,
                            "Value": outlier_value,
                            "Unit": unit,
                            "Issue Type": issue_type,
                            "TEI ID": tei_id,
                        }
                    )

    return pd.DataFrame(outliers)


def check_missing_events(tei_df, events_df):
    """Check for TEIs that have NO actual events (all has_actual_event = False)"""
    if tei_df.empty or events_df.empty:
        logging.warning("❌ Empty dataframes in check_missing_events")
        return pd.DataFrame()

    missing_events = []

    # ✅ FIXED: Get all unique TEI IDs from TEI data
    all_tei_ids = (
        set(tei_df["tei_id"].unique()) if "tei_id" in tei_df.columns else set()
    )
    logging.info(f"🔍 Newborn - Total TEI IDs: {len(all_tei_ids)}")

    # ✅ FIXED: Get TEI IDs that have ACTUAL events (has_actual_event = True)
    teis_with_actual_events = set()
    if "has_actual_event" in events_df.columns:
        actual_events = events_df[events_df["has_actual_event"] == True]
        logging.info(f"🔍 Newborn - Actual events: {len(actual_events)}")

        if not actual_events.empty:
            teis_with_actual_events = set(actual_events["tei_id"].unique())
            logging.info(
                f"🔍 Newborn - TEIs with actual events: {len(teis_with_actual_events)}"
            )
    else:
        logging.warning(
            "❌ Newborn - No 'has_actual_event' column in events dataframe!"
        )
        # If no has_actual_event column, assume all events are actual
        teis_with_actual_events = (
            set(events_df["tei_id"].unique())
            if "tei_id" in events_df.columns
            else set()
        )

    # ✅ FIXED: Find TEIs that have NO actual events
    teis_without_actual_events = all_tei_ids - teis_with_actual_events
    logging.info(
        f"🔍 Newborn - TEIs without actual events: {len(teis_without_actual_events)}"
    )

    for tei_id in teis_without_actual_events:
        first_name, last_name = get_patient_name_from_tei(tei_id, "newborn")

        # Get facility info from TEI data
        tei_row = (
            tei_df[tei_df["tei_id"] == tei_id].iloc[0] if not tei_df.empty else None
        )
        facility_name = (
            tei_row.get("orgUnit_name", "Unknown Facility")
            if tei_row is not None
            else "Unknown Facility"
        )
        region = get_region_from_facility(facility_name)

        missing_events.append(
            {
                "First Name": first_name,
                "Last Name": last_name,
                "Region": region,
                "Facility": facility_name,
                "TEI ID": tei_id,
                "Issue Type": "No Actual Events",
            }
        )

    logging.info(f"✅ Newborn - Found {len(missing_events)} missing events")
    return pd.DataFrame(missing_events)


def check_missing_data_elements(events_df):
    """Check for events with empty values in required data elements"""
    if events_df.empty:
        return pd.DataFrame()

    missing_data = []
    required_elements = set(NEWBORN_REQUIRED_ELEMENTS.keys())

    # ✅ FIXED: Only check events that have actual data (not placeholders)
    actual_events = events_df[events_df["has_actual_event"] == True]
    logging.info(
        f"🔍 Newborn - Checking {len(actual_events)} actual events for missing data elements"
    )

    # ✅ FIXED: Filter events that have required data elements with empty values
    for data_element_uid in required_elements:
        element_name = NEWBORN_REQUIRED_ELEMENTS[data_element_uid]

        # Get events for this data element
        element_events = actual_events[
            actual_events["dataElement_uid"] == data_element_uid
        ]
        logging.info(
            f"🔍 Newborn - Data element {element_name}: {len(element_events)} events"
        )

        # Find events with empty values
        empty_events = element_events[
            element_events["value"].isna() | (element_events["value"] == "")
        ]
        logging.info(
            f"🔍 Newborn - Empty events for {element_name}: {len(empty_events)}"
        )

        for _, row in empty_events.iterrows():
            tei_id = row.get("tei_id")
            first_name, last_name = get_patient_name_from_tei(tei_id, "newborn")
            facility_name = row.get("orgUnit_name", "Unknown Facility")
            region = get_region_from_facility(facility_name)

            missing_data.append(
                {
                    "First Name": first_name,
                    "Last Name": last_name,
                    "Region": region,
                    "Facility": facility_name,
                    "TEI ID": tei_id,
                    "Issue Type": "Missing Data Element",
                    "Data Element Missing": element_name,
                }
            )

    logging.info(f"✅ Newborn - Found {len(missing_data)} missing data elements")
    return pd.DataFrame(missing_data)


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

    tab1, tab2, tab3 = st.tabs(["Outliers", "Missing Events", "Missing Data"])

    with tab1:
        render_outliers_tab()
    with tab2:
        render_missing_events_tab()
    with tab3:
        render_missing_elements_tab()


def render_outliers_tab():
    """Render the outliers analysis tab"""
    events_df = st.session_state.newborn_events_df
    outliers_df = check_newborn_outliers(events_df)

    if outliers_df.empty:
        st.success("No data outliers found")
        return

    st.error(f"Found {len(outliers_df)} data outliers")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        all_regions = ["All Regions"] + sorted(outliers_df["Region"].unique())
        selected_region = st.selectbox(
            "Region", options=all_regions, index=0, key="newborn_outlier_region"
        )
    with col2:
        all_facilities = ["All Facilities"] + sorted(outliers_df["Facility"].unique())
        selected_facility = st.selectbox(
            "Facility", options=all_facilities, index=0, key="newborn_outlier_facility"
        )
    with col3:
        all_elements = ["All Data Elements"] + sorted(
            outliers_df["Data Element"].unique()
        )
        selected_element = st.selectbox(
            "Data Element", options=all_elements, index=0, key="newborn_outlier_element"
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

    # Display table
    display_df = filtered_df.copy().reset_index(drop=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))
    display_columns = [
        "No",
        "First Name",
        "Last Name",
        "Region",
        "Facility",
        "Data Element",
        "Value",
        "Unit",
        "Issue Type",
    ]

    st.dataframe(display_df[display_columns], use_container_width=True)


def render_missing_events_tab():
    """Render the missing events analysis tab"""
    tei_df = st.session_state.newborn_tei_df
    events_df = st.session_state.newborn_events_df
    missing_events_df = check_missing_events(tei_df, events_df)

    if missing_events_df.empty:
        st.success("All patients have actual events")
        return

    st.error(f"Found {len(missing_events_df)} patients with missing events")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_regions = ["All Regions"] + sorted(missing_events_df["Region"].unique())
        selected_region = st.selectbox(
            "Region", options=all_regions, index=0, key="newborn_missing_events_region"
        )
    with col2:
        all_facilities = ["All Facilities"] + sorted(
            missing_events_df["Facility"].unique()
        )
        selected_facility = st.selectbox(
            "Facility",
            options=all_facilities,
            index=0,
            key="newborn_missing_events_facility",
        )

    # Apply filters
    filtered_df = missing_events_df.copy()
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    if selected_facility != "All Facilities":
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]

    if filtered_df.empty:
        st.info("No missing events match filters")
        return

    # Display table
    display_df = filtered_df.copy().reset_index(drop=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))
    display_columns = [
        "No",
        "First Name",
        "Last Name",
        "Region",
        "Facility",
        "Issue Type",
    ]

    st.dataframe(display_df[display_columns], use_container_width=True)


def render_missing_elements_tab():
    """Render the missing data elements analysis tab"""
    events_df = st.session_state.newborn_events_df
    missing_elements_df = check_missing_data_elements(events_df)

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
            key="newborn_missing_elements_region",
        )
    with col2:
        all_facilities = ["All Facilities"] + sorted(
            missing_elements_df["Facility"].unique()
        )
        selected_facility = st.selectbox(
            "Facility",
            options=all_facilities,
            index=0,
            key="newborn_missing_elements_facility",
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

    # Display table
    display_df = filtered_df.copy().reset_index(drop=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))
    display_columns = [
        "No",
        "First Name",
        "Last Name",
        "Region",
        "Facility",
        "Data Element Missing",
    ]

    st.dataframe(display_df[display_columns], use_container_width=True)

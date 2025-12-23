# newborns_dashboard/regional_newborn.py
import streamlit as st
import pandas as pd
import logging
import time
from newborns_dashboard.kmc_coverage import compute_kmc_kpi

# IMPORT FROM NEONATAL DASH CO, NOT MATERNAL DASH_CO
from newborns_dashboard.dash_co_newborn import (
    render_trend_chart_section,
    render_comparison_chart,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
    render_kpi_tab_navigation,
    normalize_event_dates,
)

from utils.kpi_utils import clear_cache
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)

# Initialize status system
initialize_status_system()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


def count_patients_from_patient_level(df, facility_uids, org_unit_column="orgUnit"):
    """✅ Count unique patients from patient-level dataframe"""
    if df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and org_unit_column in df.columns:
        filtered_df = df[df[org_unit_column].isin(facility_uids)]
    elif facility_uids and "orgUnit" in df.columns:
        filtered_df = df[df["orgUnit"].isin(facility_uids)]
    else:
        filtered_df = df

    # Count unique patients (each row is a patient)
    return len(filtered_df)


def get_earliest_date(df, date_column):
    """Get the earliest date from a dataframe column"""
    if df.empty or date_column not in df.columns:
        return "N/A"

    try:
        # Find the earliest non-null date
        date_series = pd.to_datetime(df[date_column], errors="coerce")
        earliest_date = date_series.min()
        if pd.isna(earliest_date):
            return "N/A"
        return earliest_date.strftime("%Y-%m-%d")
    except:
        return "N/A"


def calculate_newborn_indicators(patient_df, facility_uids):
    """✅ OPTIMIZED newborn indicators calculation using patient-level data"""
    if patient_df.empty:
        return {
            "total_admitted": 0,
            "nmr": "N/A",
            "kmc_coverage_rate": 0.0,
            "kmc_cases": 0,
            "total_lbw": 0,
        }

    # ✅ Use patient-level data directly
    # Count newborns from patient-level dataframe
    total_admitted = count_patients_from_patient_level(
        patient_df, facility_uids, "orgUnit"
    )

    # Try to compute KMC coverage from patient data
    kmc_data = compute_kmc_kpi(patient_df, facility_uids)

    kmc_coverage_rate = kmc_data.get("kmc_rate", 0.0)
    kmc_cases = kmc_data.get("kmc_count", 0)
    total_lbw = kmc_data.get("total_lbw", 0)

    return {
        "total_admitted": total_admitted,
        "nmr": "N/A",  # Would need specific computation
        "kmc_coverage_rate": round(kmc_coverage_rate, 2),
        "kmc_cases": kmc_cases,
        "total_lbw": total_lbw,
    }


def get_location_display_name(selected_facilities, region_name):
    """Get the display name for location based on selection - FOLLOWING REGIONAL.PY PATTERN"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        # Join multiple facilities with comma - EXACTLY LIKE REGIONAL.PY
        return ", ".join(selected_facilities), "Facilities"


def convert_patient_to_events_format(patient_df, program_type):
    """
    Convert patient-level DataFrame back to events format for compatibility with existing functions.

    Args:
        patient_df: Patient-level DataFrame
        program_type: 'maternal' or 'newborn'

    Returns:
        DataFrame in events format
    """
    if patient_df.empty:
        return pd.DataFrame()

    logging.info(f"🔄 Converting {program_type} patient-level data to events format")

    # List to collect events
    events_list = []

    # Iterate through each patient
    for idx, row in patient_df.iterrows():
        tei_id = row.get("tei_id", "")
        org_unit = row.get("orgUnit", "")
        org_unit_name = row.get("orgUnit_name", "")

        if not tei_id:
            continue

        # Find columns that contain data elements (not metadata columns)
        metadata_cols = ["tei_id", "orgUnit", "orgUnit_name", "program_type"]

        for col in patient_df.columns:
            if col in metadata_cols:
                continue

            # Extract value
            value = row[col]

            # Skip empty values
            if pd.isna(value) or value == "":
                continue

            # Try to infer program stage and data element from column name
            program_stage_name = "Unknown"
            data_element_name = col

            # Try to extract program stage from column name
            if "_" in col:
                parts = col.split("_")
                if len(parts) >= 2:
                    # Last part might be program stage name
                    program_stage_name = parts[-1]
                    data_element_name = "_".join(parts[:-1])

            event_data = {
                "tei_id": tei_id,
                "event": f"patient_{tei_id}_{col}",
                "programStage_uid": f"patient_{program_type}",
                "programStageName": program_stage_name,
                "orgUnit": org_unit,
                "orgUnit_name": org_unit_name,
                "eventDate": "",  # Not available in patient format
                "dataElement_uid": col,
                "dataElementName": data_element_name,
                "value": str(value),
                "has_actual_event": True,
                "event_date": pd.NaT,  # Not available
                "period": "Unknown",
                "period_display": "Unknown",
                "period_sort": "999999",
            }
            events_list.append(event_data)

    events_df = pd.DataFrame(events_list)

    logging.info(
        f"✅ Converted {len(events_list)} events from {len(patient_df)} patients"
    )
    return events_df


def get_patient_data_for_dashboard(newborn_data, use_patient_level=True):
    """
    Get the appropriate data format for newborn dashboard rendering.
    Uses patient-level data if available and requested, otherwise falls back to events.

    Args:
        newborn_data: Newborn data dictionary
        use_patient_level: Whether to use patient-level data

    Returns:
        Tuple of (tei_df, enrollments_df, events_df, patient_df)
    """
    tei_df = pd.DataFrame()
    enrollments_df = pd.DataFrame()
    events_df = pd.DataFrame()
    patient_df = pd.DataFrame()

    if newborn_data:
        # ✅ Always use patient-level data from CSV
        patient_df = newborn_data.get("patients", pd.DataFrame())

        # Get empty dataframes for compatibility
        tei_df = pd.DataFrame()
        enrollments_df = pd.DataFrame()

        # Create events format from patient data if needed
        if use_patient_level and not patient_df.empty:
            logging.info("✅ Using newborn patient-level data from CSV")
            events_df = convert_patient_to_events_format(patient_df, "newborn")
            # Store patient data in session state
            st.session_state.newborn_patient_df = patient_df.copy()
        else:
            events_df = pd.DataFrame()
            st.session_state.newborn_patient_df = pd.DataFrame()

    return tei_df, enrollments_df, events_df, patient_df


def render_newborn_dashboard(
    user,
    program_uid,
    region_name,
    selected_facilities,
    facility_uids,
    view_mode,
    facility_mapping,
    facility_names,
    shared_newborn_data=None,
    use_patient_level=True,
):
    """✅ OPTIMIZED Newborn Dashboard rendering - USING PATIENT-LEVEL DATA FROM CSV"""

    # ✅ FIXED: Only run if this is the active tab
    if st.session_state.active_tab != "newborn":
        return

    logging.info("🔄 Newborn dashboard rendering - USING PATIENT-LEVEL DATA FROM CSV")

    # Use shared data if provided (from regional.py), otherwise show error
    if shared_newborn_data is not None:
        # USE THE SHARED DATA - NO DUPLICATE FETCHING
        newborn_data = shared_newborn_data
        logging.info("✅ Using shared newborn data from regional dashboard")
    else:
        # No fallback - data should come from regional dashboard
        logging.error("❌ No shared newborn data provided")
        st.error("⚠️ Newborn data not available - please refresh the dashboard")
        return

    if not newborn_data:
        st.error("⚠️ Newborn data not available")
        return

    # ✅ GET PATIENT-LEVEL DATA DIRECTLY FROM CSV
    patient_df = pd.DataFrame()
    if "patients" in newborn_data:
        patient_df = newborn_data.get("patients", pd.DataFrame())

    if patient_df.empty:
        st.error("No patient-level data available")
        return

    logging.info(f"✅ Using patient-level data from CSV: {len(patient_df)} patients")

    # ✅ DATA IS ALREADY FILTERED BY USER ACCESS - loaded from appropriate CSV file
    # The data from fetch_program_data_for_user is already filtered for the user's region/facilities

    # ✅ Normalize dates for patient data
    patient_df = normalize_event_dates(patient_df)

    # FILTER DATA BY SELECTED FACILITIES - FOLLOWING REGIONAL.PY PATTERN
    if facility_uids:
        patient_df = patient_df[patient_df["orgUnit"].isin(facility_uids)].copy()

    # Store patient data in session state
    st.session_state.newborn_patient_df = patient_df.copy()

    # ✅ Add data source indicator
    st.session_state.data_source = "csv"

    # Show connection status
    render_connection_status(patient_df, user=user)

    # Calculate total counts
    total_facilities = len(facility_mapping)

    # MAIN HEADING for Newborn program - FOLLOWING REGIONAL.PY PATTERN FOR FACILITY DISPLAY
    if selected_facilities == ["All Facilities"]:
        header_title = f"👶 Newborn Care Form - {region_name}"
        header_subtitle = f"all {len(facility_mapping)} facilities"
    elif len(selected_facilities) == 1:
        header_title = f"👶 Newborn Care Form - {selected_facilities[0]}"
        header_subtitle = "1 facility"
    else:
        header_title = "👶 Newborn Care Form - Multiple Facilities"
        header_subtitle = f"{len(selected_facilities)} facilities"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**📊 Displaying data from {header_subtitle}**")

    # ✅ Add data source indicator
    st.info(f"📁 Data loaded from CSV file: {len(patient_df)} patients")

    # ✅ ADD PROGRESS INDICATOR (same pattern as maternal)
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("---")
        st.markdown("### 📈 Preparing Newborn Dashboard...")

        progress_col1, progress_col2 = st.columns([3, 1])

        with progress_col1:
            st.markdown(
                """
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">🔄 Processing CSV Data</h4>
            <p style="margin: 5px 0; font-size: 14px;">• Computing newborn KPIs and indicators...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Generating charts and visualizations...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Preparing data tables...</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">⏱️ This may take 1-2 minutes</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with progress_col2:
            st.markdown(
                """
            <div style="text-align: center; padding: 10px;">
            <div style="font-size: 24px;">⏳</div>
            <div style="font-size: 12px; margin-top: 5px;">Processing</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        # ✅ SIMPLE FILTER CONTROLS for patient-level data
        filters = render_simple_filter_controls(
            patient_df, container=col_ctrl, context="regional_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Apply filters to patient-level data
    filtered_patients = apply_simple_filters(patient_df, filters, facility_uids)

    # Store filtered data for later use
    st.session_state["newborn_filtered_patients"] = filtered_patients.copy()
    st.session_state["last_applied_selection"] = True

    # Check for empty data
    if filtered_patients.empty:
        progress_container.empty()
        st.markdown(
            '<div class="no-data-warning">⚠️ No Newborn Care Data available for selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    # ✅ COMPUTE and STORE newborn KPIs for reuse in summary dashboard
    try:
        kmc_data = compute_kmc_kpi(filtered_patients, facility_uids)
    except Exception as e:
        logging.error(f"Error computing KMC KPI: {e}")
        kmc_data = {"kmc_rate": 0.0, "kmc_count": 0, "total_lbw": 0}

    # ✅ STORE computed newborn KPIs for reuse in summary tab
    newborn_kpis = {
        "kmc_coverage_rate": kmc_data.get("kmc_rate", 0.0),
        "kmc_cases": kmc_data.get("kmc_count", 0),
        "total_lbw": kmc_data.get("total_lbw", 0),
        "total_admitted": len(filtered_patients),
    }

    st.session_state.last_computed_newborn_kpis = newborn_kpis
    st.session_state.last_computed_newborn_timestamp = time.time()
    logging.info("✅ STORED newborn KPIs for summary dashboard reuse")

    # Get variables from filters for later use
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    # ✅ CLEAR PROGRESS INDICATOR
    progress_container.empty()

    # ---------------- KPI Trend Charts ----------------
    with col_chart:
        # Use KPI tab navigation FROM NEONATAL
        selected_kpi = render_kpi_tab_navigation()

        # Use the passed view_mode parameter - FOLLOWING REGIONAL.PY PATTERN
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - Facility Comparison - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )

            # ✅ PASS PATIENT-LEVEL DATA TO COMPARISON CHART
            render_comparison_chart(
                kpi_selection=selected_kpi,
                filtered_events=filtered_patients,  # Pass patient-level data
                comparison_mode="facility",
                display_names=facility_names,
                facility_uids=facility_uids,
                facilities_by_region=None,  # Not used in facility mode
                bg_color=bg_color,
                text_color=text_color,
                is_national=False,
            )
        else:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} Trend - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )

            # ✅ PASS PATIENT-LEVEL DATA TO TREND CHART
            render_trend_chart_section(
                selected_kpi,
                filtered_patients,  # Pass patient-level data
                facility_uids,
                facility_names,
                bg_color,
                text_color,
            )


def render_newborn_summary(
    user, region_name, selected_facilities, facility_mapping, newborn_data
):
    """Render newborn summary for the summary dashboard tab using patient-level data"""
    if not newborn_data:
        return {
            "total_admitted": 0,
            "kmc_coverage_rate": 0.0,
            "kmc_cases": 0,
            "total_lbw": 0,
            "start_date": "N/A",
        }

    # Get facility UIDs
    if selected_facilities == ["All Facilities"]:
        facility_uids = list(facility_mapping.values())
    else:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]

    # ✅ Get patient-level data directly
    patient_df = newborn_data.get("patients", pd.DataFrame())

    if patient_df.empty:
        return {
            "total_admitted": 0,
            "kmc_coverage_rate": 0.0,
            "kmc_cases": 0,
            "total_lbw": 0,
            "start_date": "N/A",
        }

    # Filter by facilities
    if facility_uids:
        patient_df = patient_df[patient_df["orgUnit"].isin(facility_uids)].copy()

    # Calculate indicators
    newborn_indicators = calculate_newborn_indicators(patient_df, facility_uids)

    # Get earliest date
    newborn_start_date = "N/A"
    # Look for date columns in patient data
    if not patient_df.empty:
        date_cols = [
            col for col in patient_df.columns if "date" in col.lower() or "Date" in col
        ]
        for date_col in date_cols:
            try:
                earliest = pd.to_datetime(patient_df[date_col], errors="coerce").min()
                if not pd.isna(earliest):
                    newborn_start_date = earliest.strftime("%Y-%m-%d")
                    break
            except:
                pass

    return {
        "total_admitted": newborn_indicators["total_admitted"],
        "kmc_coverage_rate": newborn_indicators["kmc_coverage_rate"],
        "kmc_cases": newborn_indicators["kmc_cases"],
        "total_lbw": newborn_indicators["total_lbw"],
        "start_date": newborn_start_date,
    }


# ✅ Helper function for regional.py summary dashboard
def get_newborn_patient_data(newborn_data):
    """
    Extract newborn patient-level data for summary dashboard.
    This is used by the regional.py summary dashboard.

    Args:
        newborn_data: Newborn data dictionary

    Returns:
        Dictionary with patient-level data and metadata
    """
    if not newborn_data:
        return {
            "patient_df": pd.DataFrame(),
            "tei_count": 0,
            "patient_count": 0,
            "has_patient_data": False,
        }

    # Get patient-level data if available
    patient_df = newborn_data.get("patients", pd.DataFrame())
    tei_df = newborn_data.get("tei", pd.DataFrame())

    return {
        "patient_df": patient_df,
        "tei_count": len(tei_df),
        "patient_count": len(patient_df),
        "has_patient_data": not patient_df.empty,
    }

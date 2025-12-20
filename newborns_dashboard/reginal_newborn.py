# newborns_dashboard/regional_newborn.py
import streamlit as st
import pandas as pd
import logging
import time
from newborns_dashboard.kmc_coverage import compute_kmc_kpi

# IMPORT FROM NEONATAL DASH CO, NOT MATERNAL DASH_CO
from newborns_dashboard.dash_co_newborn import (
    normalize_event_dates,
    normalize_enrollment_dates,
    render_trend_chart_section,
    render_comparison_chart,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
    render_kpi_tab_navigation,
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


def count_unique_teis_filtered(tei_df, facility_uids, org_unit_column="tei_orgUnit"):
    """Count unique TEIs from tei_df filtered by facility UIDs"""
    if tei_df.empty or not facility_uids:
        return 0

    # Filter TEI dataframe by the selected facility UIDs
    if org_unit_column in tei_df.columns:
        filtered_tei = tei_df[tei_df[org_unit_column].isin(facility_uids)]
    else:
        # Fallback to orgUnit if tei_orgUnit doesn't exist
        filtered_tei = (
            tei_df[tei_df["orgUnit"].isin(facility_uids)]
            if "orgUnit" in tei_df.columns
            else tei_df
        )

    # Count unique TEIs from the filtered dataframe
    if "tei_id" in filtered_tei.columns:
        return filtered_tei["tei_id"].nunique()
    elif "trackedEntityInstance" in filtered_tei.columns:
        return filtered_tei["trackedEntityInstance"].nunique()
    elif "tei" in filtered_tei.columns:
        return filtered_tei["tei"].nunique()
    else:
        return len(filtered_tei)


def get_earliest_date(df, date_column):
    """Get the earliest date from a dataframe column"""
    if df.empty or date_column not in df.columns:
        return "N/A"

    try:
        earliest_date = df[date_column].min()
        if pd.isna(earliest_date):
            return "N/A"
        return earliest_date.strftime("%Y-%m-%d")
    except:
        return "N/A"


def calculate_newborn_indicators(newborn_events_df, facility_uids):
    """Calculate newborn indicators using appropriate KPI functions"""
    if newborn_events_df.empty:
        return {
            "total_admitted": 0,
            "kmc_coverage_rate": 0.0,
            "kmc_cases": 0,
            "total_lbw": 0,
        }

    # Use compute_kmc_kpi for KMC coverage indicators
    kmc_data = compute_kmc_kpi(newborn_events_df, facility_uids)

    kmc_coverage_rate = kmc_data.get("kmc_rate", 0.0)
    kmc_cases = kmc_data.get("kmc_count", 0)
    total_lbw = kmc_data.get("total_lbw", 0)

    return {
        "total_admitted": 0,  # Will be set from filtered TEI count
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


def filter_data_by_facilities(data_dict, facility_uids):
    """Filter all dataframes in a data dictionary by facility UIDs - FOLLOWING REGIONAL.PY PATTERN"""
    if not data_dict or not facility_uids:
        return data_dict

    filtered_data = {}

    for key, df in data_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Filter based on available organization unit columns
            if "orgUnit" in df.columns:
                filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()
            elif "tei_orgUnit" in df.columns:
                filtered_df = df[df["tei_orgUnit"].isin(facility_uids)].copy()
            else:
                filtered_df = df.copy()  # No filtering possible
            filtered_data[key] = filtered_df
        else:
            filtered_data[key] = df

    return filtered_data


def update_facility_selection(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
):
    """Update facility selection based on current mode and selections"""
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif filter_mode == "By Region" and selected_regions:
        facility_uids, display_names = [], selected_regions
        for region in selected_regions:
            if region in facilities_by_region:
                for fac_name, fac_uid in facilities_by_region[region]:
                    facility_uids.append(fac_uid)
        comparison_mode = "region"
    elif filter_mode == "By Facility" and selected_facilities:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    else:
        # Default fallback - all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    return facility_uids, display_names, comparison_mode


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
            # This is a simplified approach - you may need to adjust based on your column naming
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
        Tuple of (tei_df, enrollments_df, events_df)
    """
    tei_df = pd.DataFrame()
    enrollments_df = pd.DataFrame()
    events_df = pd.DataFrame()

    if newborn_data:
        # Get TEI and enrollments data
        tei_df = newborn_data.get("tei", pd.DataFrame())
        enrollments_df = newborn_data.get("enrollments", pd.DataFrame())

        # Handle events data based on format preference
        if use_patient_level and "patients" in newborn_data:
            patient_df = newborn_data.get("patients", pd.DataFrame())
            if not patient_df.empty:
                logging.info("✅ Using newborn patient-level data")
                events_df = convert_patient_to_events_format(patient_df, "newborn")
                # Store patient data in session state
                st.session_state.newborn_patient_df = patient_df.copy()
                st.session_state.newborn_original_events_df = newborn_data.get(
                    "events", pd.DataFrame()
                )
            else:
                events_df = newborn_data.get("events", pd.DataFrame())
                st.session_state.newborn_patient_df = pd.DataFrame()
        else:
            events_df = newborn_data.get("events", pd.DataFrame())
            st.session_state.newborn_patient_df = pd.DataFrame()

    return tei_df, enrollments_df, events_df


def render_newborn_dashboard(
    user,
    program_uid,
    region_name,
    selected_facilities,
    facility_uids,
    view_mode,
    facility_mapping,
    facility_names,
    shared_newborn_data=None,  # ← NEW PARAMETER: Accept shared data from regional.py
    use_transformed_data=True,  # ✅ NEW: Control whether to use patient-level data
):
    """Render Newborn Care Form dashboard using shared data to avoid duplicate API calls"""

    # ✅ FIXED: Only run if this is the active tab
    if st.session_state.active_tab != "newborn":
        return

    logging.info("🔄 Newborn dashboard rendering")

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

    # ✅ UPDATED: Get data in appropriate format
    tei_df, enrollments_df, events_df = get_patient_data_for_dashboard(
        newborn_data, use_patient_level=use_transformed_data
    )

    # FILTER DATA BY SELECTED FACILITIES - FOLLOWING REGIONAL.PY PATTERN
    if facility_uids:
        # Filter each dataframe separately
        if not tei_df.empty and "tei_orgUnit" in tei_df.columns:
            tei_df = tei_df[tei_df["tei_orgUnit"].isin(facility_uids)].copy()

        if not enrollments_df.empty and "tei_orgUnit" in enrollments_df.columns:
            enrollments_df = enrollments_df[
                enrollments_df["tei_orgUnit"].isin(facility_uids)
            ].copy()

        if not events_df.empty and "orgUnit" in events_df.columns:
            events_df = events_df[events_df["orgUnit"].isin(facility_uids)].copy()

    # Normalize dates using common functions
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # Store in session state for data quality tracking
    st.session_state.newborn_events_df = events_df.copy()
    st.session_state.newborn_tei_df = tei_df.copy()

    # ✅ DEBUG: Log what we're storing
    logging.info(
        f"✅ STORED newborn data for DQ: {len(events_df)} events, {len(tei_df)} TEIs"
    )
    if not events_df.empty and "has_actual_event" in events_df.columns:
        logging.info(
            f"✅ Newborn events has_actual_event values: {events_df['has_actual_event'].value_counts().to_dict()}"
        )

    render_connection_status(events_df, user=user)

    # Calculate total counts
    total_facilities = len(facility_mapping)
    total_regions = 1  # Regional dashboard only shows one region

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

    # ✅ Add data format indicator
    if use_transformed_data and "patients" in newborn_data:
        patient_count = len(newborn_data.get("patients", pd.DataFrame()))
        st.info(f"📊 Using patient-level data format: {patient_count} patients")

    # ✅ ADD PROGRESS INDICATOR (same pattern as national newborn)
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("---")
        st.markdown("### 📈 Preparing Newborn Dashboard...")

        progress_col1, progress_col2 = st.columns([3, 1])

        with progress_col1:
            st.markdown(
                """
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">🔄 Processing Data</h4>
            <p style="margin: 5px 0; font-size: 14px;">• Computing newborn KPIs and indicators...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Generating charts and visualizations...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Preparing data tables...</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">⏱️ This may take 2-4 minutes</p>
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

        # Use simple filter controls
        filters = render_simple_filter_controls(
            events_df, container=col_ctrl, context="regional_newborn"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)

    # Store for gauge charts
    st.session_state["newborn_filtered_events"] = filtered_events.copy()

    # Check for empty data - NO KPI CARDS
    if filtered_events.empty or "event_date" not in filtered_events.columns:
        progress_container.empty()
        st.markdown(
            '<div class="no-data-warning">⚠️ No Newborn Care Data available for selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    # Get variables from filters for later use
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        progress_container.empty()
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Newborn Care Data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # ✅ COMPUTE and STORE newborn KPIs for reuse in summary dashboard
    kmc_data = compute_kmc_kpi(filtered_events, facility_uids)

    # ✅ STORE computed newborn KPIs for reuse in summary tab
    newborn_kpis = {
        "kmc_coverage_rate": kmc_data.get("kmc_rate", 0.0),
        "kmc_cases": kmc_data.get("kmc_count", 0),
        "total_lbw": kmc_data.get("total_lbw", 0),
    }

    st.session_state.last_computed_newborn_kpis = newborn_kpis
    st.session_state.last_computed_newborn_timestamp = time.time()
    logging.info("✅ STORED newborn KPIs for summary dashboard reuse")

    with col_chart:
        # Use KPI tab navigation FROM NEONATAL
        selected_kpi = render_kpi_tab_navigation()

        # Use the passed view_mode parameter - FOLLOWING REGIONAL.PY PATTERN
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - Facility Comparison - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )

            # Use render_comparison_chart FROM NEONATAL
            render_comparison_chart(
                kpi_selection=selected_kpi,
                filtered_events=filtered_events,
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

            # Use render_trend_chart_section FROM NEONATAL
            render_trend_chart_section(
                selected_kpi,
                filtered_events,
                facility_uids,
                facility_names,
                bg_color,
                text_color,
            )

    # ✅ CLEAR PROGRESS when done
    progress_container.empty()


def render_newborn_summary(
    user, region_name, selected_facilities, facility_mapping, newborn_data
):
    """Render newborn summary for the summary dashboard tab"""
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

    # ✅ UPDATED: Get data in appropriate format based on session state
    use_patient_level = st.session_state.get("use_transformed_data", True)
    tei_df, enrollments_df, events_df = get_patient_data_for_dashboard(
        newborn_data, use_patient_level=use_patient_level
    )

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids:
        # Filter each dataframe
        if not tei_df.empty and "tei_orgUnit" in tei_df.columns:
            tei_df = tei_df[tei_df["tei_orgUnit"].isin(facility_uids)].copy()

        if not enrollments_df.empty and "tei_orgUnit" in enrollments_df.columns:
            enrollments_df = enrollments_df[
                enrollments_df["tei_orgUnit"].isin(facility_uids)
            ].copy()

        if not events_df.empty and "orgUnit" in events_df.columns:
            events_df = events_df[events_df["orgUnit"].isin(facility_uids)].copy()

    # Normalize dates
    enrollments_df = normalize_enrollment_dates(enrollments_df)

    # Calculate indicators
    newborn_indicators = calculate_newborn_indicators(events_df, facility_uids)

    # Get filtered TEI count
    newborn_tei_count = count_unique_teis_filtered(tei_df, facility_uids, "tei_orgUnit")
    newborn_indicators["total_admitted"] = newborn_tei_count

    # Get earliest date
    newborn_start_date = get_earliest_date(enrollments_df, "enrollmentDate")

    return {
        "total_admitted": newborn_tei_count,
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

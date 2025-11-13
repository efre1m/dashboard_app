# newborns_dashboard/regional_newborn.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from newborns_dashboard.kmc_coverage import compute_kmc_kpi
from utils.data_service import fetch_program_data_for_user

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


# Performance optimization: Pre-load essential data with user-specific caching
@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=5)
def fetch_shared_program_data(user, program_uid):
    """Optimized shared cache for program data - user-specific"""
    if not program_uid:
        return None
    try:
        return fetch_program_data_for_user(user, program_uid)
    except Exception as e:
        logging.error(f"Error fetching data for program {program_uid}: {e}")
        return None


def get_shared_program_data_optimized(user, program_uid):
    """Optimized data loading with user-specific caching"""
    # Create user-specific session state keys
    user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    newborn_loaded_key = f"newborn_data_loaded_{user_key}"
    newborn_data_key = f"newborn_data_{user_key}"

    # Initialize session state for newborn data - user-specific
    if newborn_loaded_key not in st.session_state:
        st.session_state[newborn_loaded_key] = False
        st.session_state[newborn_data_key] = None

    # Load data if not already loaded
    if not st.session_state[newborn_loaded_key]:
        with st.spinner("🚀 Loading newborn data..."):
            try:
                st.session_state[newborn_data_key] = fetch_shared_program_data(
                    user, program_uid
                )
                st.session_state[newborn_loaded_key] = True
            except Exception as e:
                logging.error(f"Error loading newborn data: {e}")
                st.error("Newborn data loading failed. Please try refreshing.")

    return st.session_state[newborn_data_key]


def clear_newborn_cache(user=None):
    """Clear newborn data cache - user-specific"""
    if user:
        # Clear specific user cache
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        newborn_loaded_key = f"newborn_data_loaded_{user_key}"
        newborn_data_key = f"newborn_data_{user_key}"

        st.session_state[newborn_loaded_key] = False
        st.session_state[newborn_data_key] = None
    else:
        # Clear all user caches (fallback)
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith("newborn_data_loaded_") or key.startswith("newborn_data_")
        ]
        for key in keys_to_clear:
            if key.startswith("newborn_data_loaded_"):
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    clear_cache()


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


def render_newborn_dashboard(
    user,
    program_uid,
    region_name,
    selected_facilities,
    facility_uids,
    view_mode,
    facility_mapping,
    facility_names,
):
    """Render Newborn Care Form dashboard content following regional maternal dashboard pattern"""

    # Use optimized shared data loading with user-specific caching
    newborn_data = get_shared_program_data_optimized(user, program_uid)

    if not newborn_data:
        st.error("⚠️ Newborn data not available")
        return

    # FILTER DATA BY SELECTED FACILITIES - FOLLOWING REGIONAL.PY PATTERN
    if facility_uids:
        newborn_data = filter_data_by_facilities(newborn_data, facility_uids)

    # Extract dataframes efficiently
    tei_df = newborn_data.get("tei", pd.DataFrame())
    enrollments_df = newborn_data.get("enrollments", pd.DataFrame())
    events_df = newborn_data.get("events", pd.DataFrame())

    # Normalize dates using common functions
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    render_connection_status(events_df, user=user)

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
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Newborn Care Data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

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

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids:
        newborn_data = filter_data_by_facilities(newborn_data, facility_uids)

    # Extract dataframes
    tei_df = newborn_data.get("tei", pd.DataFrame())
    events_df = newborn_data.get("events", pd.DataFrame())
    enrollments_df = newborn_data.get("enrollments", pd.DataFrame())

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

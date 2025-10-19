# dashboards/newborn_dashboard.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user, program_uid):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


def count_unique_teis_filtered(tei_df, facility_uids, org_unit_column="tei_orgUnit"):
    """Count unique TEIs from tei_df filtered by facility UIDs"""
    if tei_df.empty:
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
    else:
        return 0


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


def get_location_display_name(
    filter_mode, selected_regions, selected_facilities, country_name
):
    """Get the display name for location based on selection"""
    if filter_mode == "All Facilities":
        return country_name, "Country"
    elif filter_mode == "By Region" and selected_regions:
        if len(selected_regions) == 1:
            return selected_regions[0], "Region"
        else:
            # Join multiple regions with comma
            return ", ".join(selected_regions), "Regions"
    elif filter_mode == "By Facility" and selected_facilities:
        if len(selected_facilities) == 1:
            return selected_facilities[0], "Facility"
        else:
            # Join multiple facilities with comma
            return ", ".join(selected_facilities), "Facilities"
    else:
        return country_name, "Country"


def get_current_selection_summary(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
    total_facilities,
    total_regions,
):
    """Generate selection summary without triggering full re-render"""
    if filter_mode == "All Facilities":
        return (
            f"🏥 ALL FACILITIES SELECTED<br><small>Total: {total_facilities} facilities across {total_regions} regions</small>",
            "selection-counter-all",
        )

    elif filter_mode == "By Region":
        if selected_regions:
            facilities_in_selected_regions = 0
            for region in selected_regions:
                if region in facilities_by_region:
                    facilities_in_selected_regions += len(facilities_by_region[region])

            return (
                f"🌍 REGIONS SELECTED: {len(selected_regions)}/{total_regions}<br><small>Covering {facilities_in_selected_regions} facilities</small>",
                "selection-counter-regions",
            )
        else:
            return (
                f"🌍 SELECT REGIONS<br><small>Choose from {total_regions} regions</small>",
                "selection-counter",
            )

    elif filter_mode == "By Facility":
        if selected_facilities:
            return (
                f"🏢 FACILITIES SELECTED: {len(selected_facilities)}/{total_facilities}<br><small>Across {total_regions} regions</small>",
                "selection-counter-facilities",
            )
        else:
            return (
                f"🏢 SELECT FACILITIES<br><small>Choose from {total_facilities} facilities</small>",
                "selection-counter",
            )

    # Default fallback - should never reach here, but just in case
    return (
        f"🏥 SELECTION MODE<br><small>Choose facilities to display data</small>",
        "selection-counter",
    )


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


# In newborn_dashboard.py - Update the render_newborn_dashboard function


def render_newborn_dashboard(
    user,
    program_uid,
    country_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",  # Add this parameter
):
    """Render Newborn Care Form dashboard content following maternal dashboard pattern"""

    # Fetch DHIS2 data for Newborn program
    with st.spinner(f"Fetching Newborn Care Data..."):
        try:
            dfs = fetch_cached_data(user, program_uid)
            update_last_sync_time()
        except concurrent.futures.TimeoutError:
            st.error("⚠️ DHIS2 data could not be fetched within 3 minutes.")
            return
        except requests.RequestException as e:
            st.error(f"⚠️ DHIS2 request failed: {e}")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            return

    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])
    program_info = dfs.get("program_info", {})

    # Normalize dates using common functions
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    render_connection_status(events_df, user=user)

    # Calculate total counts
    total_facilities = len(facility_mapping)
    total_regions = len(facilities_by_region.keys())

    # ---------------- Use SHARED Session State from Maternal Dashboard ----------------
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # ---------------- Use SHARED View Mode (passed as parameter) ----------------
    # view_mode is now passed from the parent component

    # ---------------- Update Facility Selection (using shared state) ----------------
    facility_uids, display_names, comparison_mode = update_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # REMOVED: View Mode Selection (now handled outside tabs)
    # Using the passed view_mode parameter instead

    # MAIN HEADING with selection summary
    selected_facilities_count = len(facility_uids)

    if comparison_mode == "facility" and "All Facilities" in display_names:
        st.markdown(
            f'<div class="main-header">👶 Newborn Care Form - {country_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from all {total_facilities} facilities**")
    elif comparison_mode == "facility" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">👶 Newborn Care Form - {display_names[0]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from 1 facility**")
    elif comparison_mode == "facility":
        st.markdown(
            f'<div class="main-header">👶 Newborn Care Form - Multiple Facilities</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities**"
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">👶 Newborn Care Form - {display_names[0]} Region</div>',
            unsafe_allow_html=True,
        )
        region_facilities_count = 0
        for region in display_names:
            if region in facilities_by_region:
                region_facilities_count += len(facilities_by_region[region])
        st.markdown(
            f"**📊 Displaying data from {region_facilities_count} facilities in 1 region**"
        )
    else:
        st.markdown(
            f'<div class="main-header">👶 Newborn Care Form - Multiple Regions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities across {len(display_names)} regions**"
        )

    # ---------------- KPI CARDS ----------------
    if events_df.empty or "event_date" not in events_df.columns:
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Newborn Care Data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Get filtered TEI count for total admitted newborns
    newborn_tei_count = count_unique_teis_filtered(tei_df, facility_uids, "tei_orgUnit")

    # REMOVED: The quick numbers/metrics display section
    # This removes the three columns with Unique Tracked Entities, Unique Enrollments, and Unique Events

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            events_df, container=col_ctrl, context="national_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()
    st.session_state["last_applied_selection"] = True

    # Get variables from filters for later use
    bg_color = filters["bg_color"]
    text_color = filters["text_color"]

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Newborn Care Data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    text_color = get_text_color(bg_color)

    with col_chart:
        # Use KPI tab navigation FROM NEONATAL
        selected_kpi = render_kpi_tab_navigation()

        # Use the passed view_mode parameter
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header">📈 {selected_kpi} - {comparison_mode.title()} Comparison - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use render_comparison_chart FROM NEONATAL
            render_comparison_chart(
                kpi_selection=selected_kpi,
                filtered_events=filtered_events,
                comparison_mode=comparison_mode,
                display_names=display_names,
                facility_uids=facility_uids,
                facilities_by_region=facilities_by_region,
                bg_color=bg_color,
                text_color=text_color,
                is_national=True,
            )
        else:
            st.markdown(
                f'<div class="section-header">📈 {selected_kpi} Trend - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use render_trend_chart_section FROM NEONATAL
            render_trend_chart_section(
                selected_kpi,
                filtered_events,
                facility_uids,
                display_names,
                bg_color,
                text_color,
            )

            st.markdown("</div>", unsafe_allow_html=True)

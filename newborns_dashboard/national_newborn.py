# dashboards/newborn_dashboard.py
import streamlit as st
import pandas as pd
import logging
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


# User-specific caching for newborn data (only used as fallback)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=5)
def fetch_cached_data(user, program_uid):
    """Fetch cached data with user-specific caching - FALLBACK ONLY"""
    import concurrent.futures

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


def filter_data_by_facilities(data_dict, facility_uids):
    """Filter all dataframes in a data dictionary by facility UIDs"""
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


def render_newborn_dashboard(
    user,
    program_uid,
    country_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",
    shared_newborn_data=None,  # ← NEW PARAMETER: Accept shared data from national.py
):
    """Render Newborn Care Form dashboard using shared data to avoid duplicate API calls"""

    # Use shared data if provided (from national.py), otherwise use fallback caching
    if shared_newborn_data is not None:
        # USE THE SHARED DATA - NO DUPLICATE FETCHING
        newborn_data = shared_newborn_data
        logging.info("✅ Using shared newborn data from national dashboard")
    else:
        # Fallback: Only fetch if called independently (not from national dashboard)
        logging.info("🔄 Fetching newborn data independently (fallback)")
        newborn_data = fetch_cached_data(user, program_uid)

    if not newborn_data:
        st.error("⚠️ Newborn data not available")
        return

    # Extract dataframes efficiently
    tei_df = newborn_data.get("tei", pd.DataFrame())
    enrollments_df = newborn_data.get("enrollments", pd.DataFrame())
    events_df = newborn_data.get("events", pd.DataFrame())

    # Normalize dates using common functions
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # Store in session state for data quality tracking
    st.session_state.newborn_events_df = events_df.copy()
    st.session_state.newborn_tei_df = tei_df.copy()

    # ---------------- Use SHARED Session State from Maternal Dashboard ----------------
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # ---------------- Update Facility Selection (using shared state) ----------------
    facility_uids, display_names, comparison_mode = update_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids and st.session_state.get("facility_filter_applied", False):
        newborn_data = filter_data_by_facilities(newborn_data, facility_uids)
        # Update dataframes after filtering
        events_df = newborn_data.get("events", pd.DataFrame())
        tei_df = newborn_data.get("tei", pd.DataFrame())

    render_connection_status(events_df, user=user)

    # Calculate total counts
    total_facilities = len(facility_mapping)
    total_regions = len(facilities_by_region.keys())

    # MAIN HEADING with selection summary
    selected_facilities_count = len(facility_uids)

    header_configs = {
        ("facility", True, 1): (
            f"👶 Newborn Care Form - {display_names[0]}",
            "1 facility",
        ),
        ("facility", True, "multiple"): (
            "👶 Newborn Care Form - Multiple Facilities",
            f"{selected_facilities_count} facilities",
        ),
        ("region", True, 1): (
            f"👶 Newborn Care Form - {display_names[0]} Region",
            f"{sum(len(facilities_by_region.get(region, [])) for region in display_names)} facilities in 1 region",
        ),
        ("region", True, "multiple"): (
            "👶 Newborn Care Form - Multiple Regions",
            f"{selected_facilities_count} facilities across {len(display_names)} regions",
        ),
    }

    key = (
        comparison_mode,
        "All Facilities" not in display_names,
        1 if len(display_names) == 1 else "multiple",
    )

    header_title, header_subtitle = header_configs.get(
        key,
        (
            f"👶 Newborn Care Form - {country_name}",
            f"all {total_facilities} facilities",
        ),
    )

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**📊 Displaying data from {header_subtitle}**")

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
    st.session_state["newborn_filtered_events"] = filtered_events.copy()

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

        # Use the passed view_mode parameter
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - {comparison_mode.title()} Comparison - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )

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
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} Trend - Newborn Care Form</div>',
                unsafe_allow_html=True,
            )

            # Use render_trend_chart_section FROM NEONATAL
            render_trend_chart_section(
                selected_kpi,
                filtered_events,
                facility_uids,
                display_names,
                bg_color,
                text_color,
            )

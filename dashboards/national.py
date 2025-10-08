import streamlit as st
from utils.dash_co import (
    render_trend_chart_section,
    render_additional_analytics,
    render_comparison_chart,
    get_text_color,
    normalize_event_dates,
    normalize_enrollment_dates,
    render_simple_filter_controls,
    apply_simple_filters,
)
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_facilities_grouped_by_region,
    get_facility_mapping_for_user,
)
from utils.kpi_utils import clear_cache
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)

# Initialize status system
initialize_status_system()


# ---------------- Robust Session State Initialization ----------------
def initialize_session_state():
    """Initialize all session state variables to prevent AttributeError"""
    session_vars = {
        "refresh_trigger": False,
        "selected_facilities": [],
        "selected_regions": [],
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
        "filter_mode": "All Facilities",  # Make sure this matches the radio options
        "filtered_events": pd.DataFrame(),
        "selection_applied": True,
        "cached_events_data": None,
        "cached_enrollments_data": None,
        "cached_tei_data": None,
        "last_applied_selection": "All Facilities",
        "kpi_cache": {},
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state at the very beginning
initialize_session_state()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# ---------------- Cache Wrappers ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_facilities(user):
    """Cache facility data to avoid repeated API calls"""
    facilities_by_region = get_facilities_grouped_by_region(user)
    facility_mapping = get_facility_mapping_for_user(user)
    return facilities_by_region, facility_mapping


# ---------------- Data Processing Functions ----------------
def process_and_cache_data(dfs):
    """Process data and cache in session state to avoid reprocessing"""
    if (
        st.session_state.cached_events_data is not None
        and st.session_state.cached_enrollments_data is not None
        and st.session_state.cached_tei_data is not None
    ):
        return

    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())

    # Normalize dates
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # Cache processed data
    st.session_state.cached_tei_data = tei_df
    st.session_state.cached_enrollments_data = enrollments_df
    st.session_state.cached_events_data = events_df


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


# ---------------- Page Rendering ----------------
def render():
    st.set_page_config(
        page_title="National Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Re-initialize session state for safety
    initialize_session_state()

    # Load CSS files
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    try:
        with open("utils/national.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Sidebar user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    country_name = user.get("country_name", "Unknown country")

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🌍 Country: {country_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh Data Button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        clear_cache()
        st.session_state.cached_events_data = None
        st.session_state.cached_enrollments_data = None
        st.session_state.cached_tei_data = None
        st.session_state.refresh_trigger = not st.session_state.refresh_trigger
        st.session_state.selection_applied = True
        st.rerun()

    # Fetch DHIS2 data (only if not cached or refresh needed)
    if (
        st.session_state.cached_events_data is None or st.session_state.refresh_trigger
    ):  # REMOVED: or should_refresh_data()

        with st.spinner("Fetching national maternal data..."):
            try:
                dfs = fetch_cached_data(user)  # This automatically uses 10-minute cache
                process_and_cache_data(dfs)
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

    # Use cached data
    tei_df = st.session_state.cached_tei_data
    enrollments_df = st.session_state.cached_enrollments_data
    events_df = st.session_state.cached_events_data

    render_connection_status(
        st.session_state.get("cached_events_data", pd.DataFrame()), user=user
    )

    # Get facility data (cached) - with error handling
    try:
        facilities_by_region, facility_mapping = get_cached_facilities(user)

        # Flatten all facility names
        all_facility_names = [
            f for region, facs in facilities_by_region.items() for f, _ in facs
        ]
        all_region_names = list(facilities_by_region.keys())

        # Calculate total counts
        total_facilities = len(facility_mapping)
        total_regions = len(all_region_names)

    except Exception as e:
        st.error(f"⚠️ Error loading facility data: {e}")
        # Set default values to prevent further errors
        facilities_by_region = {}
        facility_mapping = {}
        all_facility_names = []
        all_region_names = []
        total_facilities = 0
        total_regions = 0

    # ---------------- Selection Counter Display ----------------
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # Add safe unpacking with default values
    try:
        summary_text, css_class = get_current_selection_summary(
            filter_mode,
            selected_regions,
            selected_facilities,
            facilities_by_region,
            facility_mapping,
            total_facilities,
            total_regions,
        )
    except Exception as e:
        # Fallback values if the function fails
        summary_text = (
            "🏥 SELECTION MODE<br><small>Choose facilities to display data</small>"
        )
        css_class = "selection-counter"

    st.sidebar.markdown(
        f'<div class="selection-counter {css_class}">{summary_text}</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Mode Selection ----------------
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Facility Selection Mode</p>',
        unsafe_allow_html=True,
    )

    # Fix the radio index calculation with safe fallback
    radio_options = ["All Facilities", "By Region", "By Facility"]

    # Safely get the index - handle cases where filter_mode might have old values
    try:
        current_index = radio_options.index(st.session_state.filter_mode)
    except ValueError:
        # If filter_mode contains an old value like 'facility', reset to default
        current_index = 0
        st.session_state.filter_mode = "All Facilities"

    new_filter_mode = st.sidebar.radio(
        "Select facilities by:",
        radio_options,
        index=current_index,
        key="mode_radio",
    )

    # Update mode without resetting selections
    if new_filter_mode != st.session_state.filter_mode:
        st.session_state.filter_mode = new_filter_mode
        # Don't reset selections - preserve them across mode changes

    # ---------------- Selection Form ----------------
    with st.sidebar.form("selection_form"):
        temp_selected_regions = st.session_state.selected_regions.copy()
        temp_selected_facilities = st.session_state.selected_facilities.copy()

        if st.session_state.filter_mode == "By Region":
            st.markdown("**🌍 Select Regions**")

            # Multi-select dropdown for regions with facility counts
            region_options = {
                f"{region} ({len(facilities_by_region.get(region, []))} facilities)": region
                for region in all_region_names
            }

            selected_region_labels = st.multiselect(
                "Choose regions:",
                options=list(region_options.keys()),
                default=[
                    label
                    for label in region_options.keys()
                    if region_options[label] in st.session_state.selected_regions
                ],
                help="Select one or more regions",
                key="region_multiselect",
            )

            # Convert back to region names
            temp_selected_regions = [
                region_options[label] for label in selected_region_labels
            ]

        elif st.session_state.filter_mode == "By Facility":
            st.markdown("**🏢 Select Facilities (grouped by region)**")

            for region_name, facilities in facilities_by_region.items():
                total_count = len(facilities)
                selected_count = sum(
                    1 for fac, _ in facilities if fac in temp_selected_facilities
                )

                # Determine selection status
                if selected_count == 0:
                    header_class = "region-header-none-selected"
                    icon = "○"
                elif selected_count == total_count:
                    header_class = "region-header-fully-selected"
                    icon = "✅"
                else:
                    header_class = "region-header-partially-selected"
                    icon = "⚠️"

                with st.expander(
                    f"{icon} {region_name} ({selected_count}/{total_count} selected)",
                    expanded=False,
                ):
                    # Select all checkbox
                    all_selected_in_region = all(
                        fac in temp_selected_facilities for fac, _ in facilities
                    )
                    select_all_box = st.checkbox(
                        f"Select all in {region_name}",
                        value=all_selected_in_region,
                        key=f"select_all_{region_name}",
                    )

                    # Handle select all logic
                    if select_all_box and not all_selected_in_region:
                        for fac_name, _ in facilities:
                            if fac_name not in temp_selected_facilities:
                                temp_selected_facilities.append(fac_name)
                    elif not select_all_box and all_selected_in_region:
                        for fac_name, _ in facilities:
                            if fac_name in temp_selected_facilities:
                                temp_selected_facilities.remove(fac_name)

                    # Individual facility checkboxes
                    for fac_name, _ in facilities:
                        fac_checked = fac_name in temp_selected_facilities
                        fac_checked = st.checkbox(
                            fac_name,
                            value=fac_checked,
                            key=f"fac_{region_name}_{fac_name}",
                        )
                        if fac_checked and fac_name not in temp_selected_facilities:
                            temp_selected_facilities.append(fac_name)
                        elif not fac_checked and fac_name in temp_selected_facilities:
                            temp_selected_facilities.remove(fac_name)

        else:  # All Facilities mode
            st.markdown("**🏥 All Facilities Mode**")

        selection_submitted = st.form_submit_button("✅ Apply Selection")

        if selection_submitted:
            # Update selections and trigger data display
            st.session_state.selected_regions = temp_selected_regions
            st.session_state.selected_facilities = temp_selected_facilities
            st.session_state.selection_applied = True
            st.rerun()

    # ---------------- Update Facility Selection ----------------
    facility_uids, display_names, comparison_mode = update_facility_selection(
        st.session_state.filter_mode,
        st.session_state.selected_regions,
        st.session_state.selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if (comparison_mode == "facility" and len(display_names) > 1) or (
        comparison_mode == "region" and len(display_names) > 1
    ):
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Comparison View"],
            index=0,
            help="Compare trends across multiple facilities or regions",
            key="view_mode_radio",
        )

    # MAIN HEADING with selection summary
    selected_facilities_count = len(facility_uids)
    if comparison_mode == "facility" and "All Facilities" in display_names:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {country_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from all {total_facilities} facilities**")
    elif comparison_mode == "facility" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from 1 facility**")
    elif comparison_mode == "facility":
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Facilities</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities**"
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]} Region</div>',
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
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Regions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities across {len(display_names)} regions**"
        )

    # ---------------- Data Display Logic ----------------
    if not st.session_state.get("selection_applied", False):
        # Show previous data if available
        if (
            "filtered_events" in st.session_state
            and not st.session_state.filtered_events.empty
            and st.session_state.get("last_applied_selection")
        ):
            filtered_events = st.session_state.filtered_events
            display_name = country_name
            user_id = str(user.get("id", user.get("username", "default_user")))

            # Render KPIs with cached data
            render_kpi_cards(
                filtered_events,
                st.session_state.current_facility_uids,
                display_name,
                user_id=user_id,
            )

            # Render charts with cached data
            col_chart, col_ctrl = st.columns([3, 1])
            with col_ctrl:
                st.markdown('<div class="filter-box">', unsafe_allow_html=True)
                filters = render_simple_filter_controls(
                    filtered_events, container=col_ctrl
                )
                st.markdown("</div>", unsafe_allow_html=True)

            kpi_selection = filters["kpi_selection"]
            bg_color = filters["bg_color"]
            text_color = filters["text_color"]

            with col_chart:
                if view_mode == "Comparison View" and len(display_names) > 1:
                    st.markdown(
                        f'<div class="section-header">📈 {kpi_selection} - {comparison_mode.title()} Comparison</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                    render_comparison_chart(
                        kpi_selection=kpi_selection,
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
                        f'<div class="section-header">📈 {kpi_selection} Trend</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                    render_trend_chart_section(
                        kpi_selection,
                        filtered_events,
                        facility_uids,
                        display_names,
                        bg_color,
                        text_color,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                render_additional_analytics(
                    kpi_selection, filtered_events, facility_uids, bg_color, text_color
                )
        return

    # ---------------- KPI CARDS (Only when selection applied) ----------------
    if events_df.empty or "event_date" not in events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # 🔒 Always national view
    display_name = country_name
    user_id = str(user.get("id", user.get("username", "default_user")))
    all_facility_uids = list(facility_mapping.values())

    render_kpi_cards(
        events_df,
        all_facility_uids,
        display_name,
        user_id=user_id,
    )

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(events_df, container=col_ctrl)
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()
    st.session_state["last_applied_selection"] = True

    kpi_selection = filters["kpi_selection"]
    bg_color = filters["bg_color"]
    text_color = filters["text_color"]

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    text_color = get_text_color(bg_color)

    with col_chart:
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header">📈 {kpi_selection} - {comparison_mode.title()} Comparison</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            render_comparison_chart(
                kpi_selection=kpi_selection,
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
                f'<div class="section-header">📈 {kpi_selection} Trend</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            render_trend_chart_section(
                kpi_selection,
                filtered_events,
                facility_uids,
                display_names,
                bg_color,
                text_color,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        render_additional_analytics(
            kpi_selection, filtered_events, facility_uids, bg_color, text_color
        )

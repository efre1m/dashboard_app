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


# ---------------- Robust Session State Initialization ----------------
def initialize_session_state():
    """Initialize all session state variables to prevent AttributeError"""
    session_vars = {
        "refresh_trigger": False,
        "all_facilities_checkbox": False,
        "all_regions_checkbox": False,
        "selected_facilities": [],
        "selected_regions": [],
        "expanded_regions": {},
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
        "filter_mode": "All Facilities",
        "filtered_events": pd.DataFrame(),
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state at the very beginning
initialize_session_state()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# ---------------- Cache Wrapper ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)


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
    except Exception as e:
        st.warning(f"Facility CSS file not found: {e}")

    try:
        with open("utils/national.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"National CSS file not found: {e}")

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

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]
        st.rerun()

    # Fetch DHIS2 data
    with st.spinner("Fetching national maternal data..."):
        try:
            dfs = fetch_cached_data(user)
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

    # Normalize dates
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    copied_events_df = normalize_event_dates(events_df)

    # ---------------- Facility and Region Filter ----------------
    facilities_by_region = get_facilities_grouped_by_region(user)
    facility_mapping = get_facility_mapping_for_user(user)

    # Flatten all facility names
    all_facility_names = [
        f for region, facs in facilities_by_region.items() for f, _ in facs
    ]
    all_region_names = list(facilities_by_region.keys())

    # Calculate total counts
    total_facilities = len(facility_mapping)
    total_regions = len(all_region_names)

    # ---------------- Selection Counter Display ----------------
    current_selected_facilities_count = len(st.session_state.selected_facilities)
    current_selected_regions_count = len(st.session_state.selected_regions)
    filter_mode = st.session_state.get("filter_mode", "All Facilities")

    if filter_mode == "All Facilities":
        st.sidebar.markdown(
            f"""
            <div class="selection-counter selection-counter-all">
                🏥 ALL FACILITIES SELECTED<br>
                <small>Total: {total_facilities} facilities across {total_regions} regions</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif filter_mode == "By Region":
        if current_selected_regions_count > 0:
            facilities_in_selected_regions = 0
            for region in st.session_state.selected_regions:
                if region in facilities_by_region:
                    facilities_in_selected_regions += len(facilities_by_region[region])

            st.sidebar.markdown(
                f"""
                <div class="selection-counter selection-counter-regions">
                    🌍 REGIONS SELECTED: {current_selected_regions_count}/{total_regions}<br>
                    <small>Covering {facilities_in_selected_regions} facilities</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                f"""
                <div class="selection-counter">
                    🌍 SELECT REGIONS<br>
                    <small>Choose from {total_regions} regions</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif filter_mode == "By Facility":
        if current_selected_facilities_count > 0:
            st.sidebar.markdown(
                f"""
                <div class="selection-counter selection-counter-facilities">
                    🏢 FACILITIES SELECTED: {current_selected_facilities_count}/{total_facilities}<br>
                    <small>Across {total_regions} regions</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                f"""
                <div class="selection-counter">
                    🏢 SELECT FACILITIES<br>
                    <small>Choose from {total_facilities} facilities</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- Sidebar Filter ----------------
    with st.sidebar.form("mode_selection_form"):
        st.markdown(
            '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Facility Selection Mode</p>',
            unsafe_allow_html=True,
        )

        filter_mode = st.radio(
            "Select facilities by:",
            ["All Facilities", "By Region", "By Facility"],
            index=0,
        )

        submitted_mode = st.form_submit_button("✅ Apply Mode")
        if submitted_mode:
            st.session_state.selected_regions = []
            st.session_state.selected_facilities = []
            st.session_state.filter_mode = filter_mode
            st.rerun()

    filter_mode = st.session_state.get("filter_mode", "All Facilities")

    # ---- Form 2: Selections (Regions or Facilities) ----
    with st.sidebar.form("selection_form"):
        if filter_mode == "By Region":
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
            )

            # Convert back to region names
            temp_selected_regions = [
                region_options[label] for label in selected_region_labels
            ]

            submitted_selection = st.form_submit_button("✅ Apply Selection")
            if submitted_selection:
                st.session_state.selected_regions = temp_selected_regions
                st.session_state.selected_facilities = []
                st.rerun()

        elif filter_mode == "By Facility":
            st.markdown("**🏢 Select Facilities (grouped by region)**")
            temp_selected_facilities = st.session_state.selected_facilities.copy()

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
                    st.markdown(
                        f"""<div class="{header_class}">{region_name} - {selected_count}/{total_count} selected</div>""",
                        unsafe_allow_html=True,
                    )

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

            submitted_selection = st.form_submit_button("✅ Apply Selection")
            if submitted_selection:
                st.session_state.selected_facilities = temp_selected_facilities
                st.session_state.selected_regions = []
                st.rerun()

    # ---------------- Update session_state for dashboard ----------------
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
        st.session_state.selected_regions = []
        st.session_state.selected_facilities = []
    elif st.session_state.selected_regions:
        facility_uids, display_names = [], st.session_state.selected_regions
        for region in st.session_state.selected_regions:
            if region in facilities_by_region:
                for fac_name, fac_uid in facilities_by_region[region]:
                    facility_uids.append(fac_uid)
        comparison_mode = "region"
    elif st.session_state.selected_facilities:
        facility_uids = [
            facility_mapping[f]
            for f in st.session_state.selected_facilities
            if f in facility_mapping
        ]
        display_names = st.session_state.selected_facilities
        comparison_mode = "facility"
    else:
        # Default fallback - all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    facility_uids = st.session_state.current_facility_uids
    display_names = st.session_state.current_display_names
    comparison_mode = st.session_state.current_comparison_mode

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

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
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
        copied_events_df,
        all_facility_uids,
        display_name,
        user_id=user_id,
    )

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(copied_events_df, container=col_ctrl)
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(copied_events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()

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

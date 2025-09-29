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

    # Load both CSS files - facility.css first, then national.css for overrides
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
        # Fallback to basic national styling
        st.markdown(
            """
            <style>
            /* National-specific fallback styles */
            .sidebar .sidebar-content {
                background: linear-gradient(135deg, #1a5fb4 0%, #1c71d8 100%);
                color: white;
            }
            .user-info {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .section-header {
                color: white !important;
                font-weight: 700;
                font-size: 18px;
                margin: 20px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            }
            .stCheckbox [data-baseweb="checkbox"]:checked {
                background-color: #ffa348;
                border-color: #ffa348;
            }
            .stExpander {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .stExpander summary {
                color: white !important;
                font-weight: 600;
                padding: 12px 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            /* Selection status styles */
            .region-header-fully-selected {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
                color: white !important;
                border-left: 4px solid #2E7D32 !important;
            }
            .region-header-partially-selected {
                background: linear-gradient(135deg, #FFC107 0%, #FFB300 100%) !important;
                color: black !important;
                border-left: 4px solid #FF8F00 !important;
            }
            .region-header-none-selected {
                background: rgba(255, 255, 255, 0.1) !important;
                color: white !important;
                border-left: 4px solid rgba(255, 255, 255, 0.3) !important;
            }
            /* Fix for button visibility */
            .stButton > button {
                color: black !important;
                border: 1px solid #ccc !important;
                background-color: #f0f2f6 !important;
            }
            /* Selection counter styles */
            .selection-counter {
                background: linear-gradient(135deg, #1a5fb4 0%, #1c71d8 100%);
                color: white;
                padding: 10px 15px;
                border-radius: 8px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                font-weight: 600;
            }
            .selection-counter-all {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            }
            .selection-counter-regions {
                background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
            }
            .selection-counter-facilities {
                background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

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

    # Calculate current selection counts
    current_selected_facilities_count = len(st.session_state.selected_facilities)
    current_selected_regions_count = len(st.session_state.selected_regions)

    # Display selection counter based on current mode
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
            # Calculate total facilities in selected regions
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

    # ---- Form 1: Mode selection ----
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
            # Clear previous selections when switching modes
            st.session_state.selected_regions = []
            st.session_state.selected_facilities = []
            st.session_state.filter_mode = filter_mode
            st.rerun()  # Force immediate update

    # Use previously selected mode
    filter_mode = st.session_state.get("filter_mode", "All Facilities")

    # ---- Form 2: Selections (Regions or Facilities) ----
    with st.sidebar.form("selection_form"):
        if filter_mode == "By Region":
            st.markdown("**🌍 Select Regions**")
            temp_selected_regions = st.session_state.selected_regions.copy()

            # All Regions checkbox
            all_regions_selected = len(temp_selected_regions) == len(all_region_names)
            all_regions_box = st.checkbox(
                "All Regions", value=all_regions_selected, key="all_regions_checkbox"
            )

            if all_regions_box:
                temp_selected_regions = all_region_names.copy()
            elif all_regions_selected and not all_regions_box:
                temp_selected_regions = []

            # Individual region checkboxes with facility counts
            for region in all_region_names:
                region_facility_count = len(facilities_by_region.get(region, []))
                checked = region in temp_selected_regions
                checked = st.checkbox(
                    f"{region} ({region_facility_count} facilities)",
                    value=checked,
                    key=f"region_{region}",
                )
                if checked and region not in temp_selected_regions:
                    temp_selected_regions.append(region)
                elif not checked and region in temp_selected_regions:
                    temp_selected_regions.remove(region)

            submitted_selection = st.form_submit_button("✅ Apply Selection")
            if submitted_selection:
                st.session_state.selected_regions = temp_selected_regions
                st.session_state.selected_facilities = (
                    []
                )  # clear facilities when regions selected
                st.rerun()  # Force immediate UI update

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

                # Create expander with custom header
                with st.expander(
                    f"{icon} {region_name} ({selected_count}/{total_count} selected)",
                    expanded=False,
                ):

                    # Apply header styling
                    st.markdown(
                        f"""<div class="{header_class}">{region_name} - {selected_count}/{total_count} selected</div>""",
                        unsafe_allow_html=True,
                    )

                    # Select all checkbox with immediate state handling
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
                st.session_state.selected_regions = (
                    []
                )  # clear regions when facilities selected
                st.rerun()  # Force immediate UI update

    # ---------------- Update session_state for dashboard ----------------
    # Handle "All Facilities" mode - always use all facilities regardless of previous selections
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
        # Clear any previous selections
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

    # ---------------- Use session_state in rest of dashboard ----------------
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
    if (
        comparison_mode == "facility"
        and "All Facilities" in st.session_state.current_display_names
    ):
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
        # Calculate facilities in this region
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

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(user.get("id", user.get("username", "default_user")))

    all_facility_uids = list(facility_mapping.values())  # all facilities in the DB
    render_kpi_cards(
        copied_events_df,  # full dataset
        all_facility_uids,  # force ALL facilities
        display_name,  # ✅ fixed to national label
        user_id=user_id,
    )

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        # Use simple filter controls
        filters = render_simple_filter_controls(copied_events_df, container=col_ctrl)

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(copied_events_df, filters, facility_uids)

    # Store for gauge charts
    st.session_state["filtered_events"] = filtered_events.copy()

    # Get variables from filters for later use
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

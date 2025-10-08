# dashboards/regional.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_for_user,
    get_facility_mapping_for_user,
)
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
)
from utils.kpi_utils import clear_cache
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)

initialize_status_system()


def initialize_session_state():
    """Initialize all session state variables to prevent AttributeError"""
    session_vars = {
        "refresh_trigger": False,
        "selected_facilities": [],
        "selected_regions": [],
        "current_facility_uids": [],
        "current_display_names": [],
        "current_comparison_mode": "facility",
        "filter_mode": "facility",
        "filtered_events": pd.DataFrame(),
        "selection_applied": True,
        "cached_events_data": None,
        "cached_enrollments_data": None,
        "cached_tei_data": None,
        "last_applied_selection": None,
        "kpi_cache": {},
        "selected_program_uid": None,
        "selected_program_name": "Maternal Inpatient Data",
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state at the very beginning
initialize_session_state()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 600  # 10 minutes


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user, program_uid):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


def render_program_selector():
    """Render program selection dropdown in sidebar"""
    programs = get_all_programs()

    if not programs:
        st.sidebar.error("No programs found in database")
        return None

    # Create display names for dropdown
    program_options = {p["program_name"]: p["program_uid"] for p in programs}

    # Get current selection
    current_program_name = st.session_state.get(
        "selected_program_name", "Maternal Inpatient Data"
    )

    # Add CSS to make the label white
    st.sidebar.markdown(
        """
        <style>
        .program-selector-label {
            color: white !important;
            font-weight: 600;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<p class="program-selector-label">📋 Select Program</p>',
        unsafe_allow_html=True,
    )

    # Program selector
    selected_program_name = st.sidebar.selectbox(
        " ",  # Empty label since we're using the styled label above
        options=list(program_options.keys()),
        index=(
            list(program_options.keys()).index(current_program_name)
            if current_program_name in program_options
            else 0
        ),
        key="program_selector",
        label_visibility="collapsed",  # Hide the default label
    )

    selected_program_uid = program_options[selected_program_name]

    # Update session state if program changed
    if selected_program_uid != st.session_state.get(
        "selected_program_uid"
    ) or selected_program_name != st.session_state.get("selected_program_name"):
        st.session_state.selected_program_uid = selected_program_uid
        st.session_state.selected_program_name = selected_program_name
        st.session_state.refresh_trigger = not st.session_state.refresh_trigger
        st.rerun()

    return selected_program_uid


def render_newborn_dashboard(region_name, facilities, facility_mapping):
    """Render Newborn Care Form dashboard content"""
    st.markdown(
        f'<div class="main-header">👶 Newborn Care Dashboard - {region_name}</div>',
        unsafe_allow_html=True,
    )

    st.info("🚧 **Newborn Care Dashboard Coming Soon!**")
    st.markdown(
        """
    We're working on building the Newborn Care dashboard with specialized KPIs and visualizations.
    
    **Features coming soon:**
    - Newborn admission metrics
    - Birth weight tracking
    - Feeding status monitoring
    - Vaccination coverage
    - Growth monitoring charts
    
    In the meantime, please use the **Maternal Inpatient Data** program for maternal health analytics.
    """
    )


def render_maternal_dashboard(
    user, program_uid, region_name, facilities, facility_mapping
):
    """Render Maternal Inpatient Data dashboard content"""
    # Fetch DHIS2 data for Maternal program
    with st.spinner(f"Fetching Maternal Inpatient Data..."):
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
    copied_events_df = normalize_event_dates(events_df)

    render_connection_status(copied_events_df, user=user)

    # ---------------- Facility Filter ----------------
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Select Facilities</p>',
        unsafe_allow_html=True,
    )

    # Default to all facilities
    default_facilities = ["All Facilities"]

    selected_facilities = st.sidebar.multiselect(
        " ",
        ["All Facilities"] + facilities,
        default=default_facilities,
        key="facility_selector",
        label_visibility="collapsed",
    )

    # 👇 Dynamic count below the dropdown
    total_facilities = len(facilities)
    if selected_facilities == ["All Facilities"]:
        display_text = f"Selected: All ({total_facilities})"
    else:
        display_text = f"Selected: {len(selected_facilities)} / {total_facilities}"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: -10px;'>{display_text}</p>",
        unsafe_allow_html=True,
    )

    # Handle "All Facilities" selection logic
    if "All Facilities" in selected_facilities:
        if len(selected_facilities) > 1:
            # If "All Facilities" is selected with others, remove "All Facilities"
            selected_facilities = [
                f for f in selected_facilities if f != "All Facilities"
            ]
        else:
            # Only "All Facilities" is selected
            selected_facilities = ["All Facilities"]

    # Get the facility UIDs for selected facilities
    facility_uids = None
    facility_names = None
    if selected_facilities != ["All Facilities"]:
        facility_uids = [
            facility_mapping[facility]
            for facility in selected_facilities
            if facility in facility_mapping
        ]
        facility_names = selected_facilities

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if selected_facilities != ["All Facilities"] and len(selected_facilities) > 1:
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Facility Comparison"],
            index=0,
            help="Compare trends across multiple facilities",
        )

    # MAIN HEADING for Maternal program
    if selected_facilities == ["All Facilities"]:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Inpatient Data - {region_name}</div>',
            unsafe_allow_html=True,
        )
    elif len(selected_facilities) == 1:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Inpatient Data - {selected_facilities[0]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Inpatient Data - Multiple Facilities ({len(selected_facilities)})</div>',
            unsafe_allow_html=True,
        )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Maternal Inpatient Data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # 🔒 Always show REGIONAL KPI values (ignore facility filters)
    all_facility_uids = list(facility_mapping.values())  # all facilities in this region

    # Use the REGION name for display (not country)
    display_name = region_name

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(user.get("id", user.get("username", "default_user")))

    # Render KPI cards (locked to regional level)
    render_kpi_cards(
        copied_events_df,  # full dataset (already unfiltered copy)
        all_facility_uids,  # force ALL facilities in region
        display_name,  # always show region name
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
            f'<div class="no-data-warning">⚠️ No Maternal Inpatient Data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    text_color = get_text_color(bg_color)

    with col_chart:
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header">📈 {kpi_selection} - Facility Comparison - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use common comparison chart function
            render_comparison_chart(
                kpi_selection=kpi_selection,
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
                f'<div class="section-header">📈 {kpi_selection} Trend - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use common trend chart function
            render_trend_chart_section(
                kpi_selection,
                filtered_events,
                facility_uids,
                facility_names,
                bg_color,
                text_color,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Use common additional analytics function
        render_additional_analytics(
            kpi_selection, filtered_events, facility_uids, bg_color, text_color
        )


def render():
    st.set_page_config(
        page_title="IMNID Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

    # Load CSS if available
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Sidebar user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    region_name = user.get("region_name", "Unknown Region")

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🗺️ Region: {region_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Program selection
    program_uid = render_program_selector()
    if not program_uid:
        return

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        clear_cache()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # ---------------- Get Facilities Data (SAME FOR BOTH PROGRAMS) ----------------
    db_facilities = get_facilities_for_user(user)
    facilities = [facility[0] for facility in db_facilities]
    facility_mapping = get_facility_mapping_for_user(user)

    # ✅ CLEAR PROGRAM SELECTION LOGIC
    selected_program_name = st.session_state.selected_program_name

    if selected_program_name == "Newborn Care Form":
        # GROUP 1: Newborn Care Form Content
        render_newborn_dashboard(region_name, facilities, facility_mapping)

    elif selected_program_name == "Maternal Inpatient Data":
        # GROUP 2: Maternal Inpatient Data Content
        render_maternal_dashboard(
            user, program_uid, region_name, facilities, facility_mapping
        )

    else:
        # Fallback: Show Maternal dashboard
        st.warning(
            f"Unknown program: {selected_program_name}. Showing Maternal Inpatient Data."
        )
        render_maternal_dashboard(
            user, program_uid, region_name, facilities, facility_mapping
        )

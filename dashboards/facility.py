# dashboards/facility.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.queries import get_all_programs
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    render_trend_chart_section,
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
CACHE_TTL = 600  # 30 minutes


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user, program_uid):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


def render_newborn_dashboard(facility_name):
    """Render Newborn Care Form dashboard content"""
    st.markdown(
        f'<div class="main-header">👶 Newborn Care Dashboard - {facility_name}</div>',
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


def render_maternal_dashboard(user, program_uid, facility_name, facility_uid):
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

    # Filter data to only show this facility's data
    if facility_uid and not copied_events_df.empty:
        copied_events_df = copied_events_df[copied_events_df["orgUnit"] == facility_uid]

    render_connection_status(copied_events_df, user=user)

    # MAIN HEADING for Maternal program
    st.markdown(
        f'<div class="main-header">🤰 Maternal Inpatient Data - {facility_name}</div>',
        unsafe_allow_html=True,
    )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            f'<div class="no-data-warning">⚠️ No Maternal Inpatient Data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Pass user_id into KPI card renderer
    user_id = str(user.get("id", user.get("username", "Unknown User")))

    # Use single facility UID (not list) for facility-level view
    render_kpi_cards(
        copied_events_df,
        facility_uid,  # Single facility UID (not list)
        facility_name,
        user_id=user_id,
    )

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        # Use simple filter controls
        filters = render_simple_filter_controls(copied_events_df, container=col_ctrl)

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters with single facility UID
    filtered_events = apply_simple_filters(copied_events_df, filters, facility_uid)

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
        st.markdown(
            f'<div class="section-header">📈 {kpi_selection} Trend - Maternal Inpatient Data</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Use common trend chart function with single facility
        render_trend_chart_section(
            kpi_selection,
            filtered_events,
            facility_uid,  # Single facility UID
            facility_name,  # Single facility name
            bg_color,
            text_color,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Use common additional analytics function
        render_additional_analytics(
            kpi_selection, filtered_events, facility_uid, bg_color, text_color
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
    facility_name = user.get("facility_name", "Unknown Facility")
    facility_uid = user.get("facility_uid")

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🏥 Facility: {facility_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        clear_cache()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # Get programs for UID mapping
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    # CREATE PROFESSIONAL TABS IN MAIN AREA
    tab1, tab2 = st.tabs(["🤰 **Maternal Inpatient Data**", "👶 **Newborn Care Form**"])

    with tab1:
        # GROUP 1: Maternal Inpatient Data Content
        maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
        if maternal_program_uid:
            render_maternal_dashboard(
                user, maternal_program_uid, facility_name, facility_uid
            )
        else:
            st.error("Maternal Inpatient Data program not found")

    with tab2:
        # GROUP 2: Newborn Care Form Content
        newborn_program_uid = program_uid_map.get("Newborn Care Form")
        if newborn_program_uid:
            render_newborn_dashboard(facility_name)
        else:
            st.error("Newborn Care Form program not found")

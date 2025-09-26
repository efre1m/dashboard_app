import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    get_kpi_selection,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
    apply_date_filters,
    KPI_OPTIONS,
)

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
        page_title="Maternal Health Dashboard",
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
    facility_uid = user.get("facility_uid")  # Get facility UID from user session

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

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # Fetch DHIS2 data
    with st.spinner("Fetching maternal data..."):
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

    # Normalize dates using common functions
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    copied_events_df = normalize_event_dates(events_df)

    # Filter data to only show this facility's data
    if facility_uid and not copied_events_df.empty:
        copied_events_df = copied_events_df[copied_events_df["orgUnit"] == facility_uid]

    # MAIN HEADING
    st.markdown(
        f'<div class="main-header">🏥 Maternal Health Dashboard - {facility_name}</div>',
        unsafe_allow_html=True,
    )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(
        user.get("id", username)
    )  # Prefer numeric ID if available, fallback to username

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

        # Use common KPI selection
        kpi_selection = get_kpi_selection()

        # Build a minimal df for date range defaults
        _df_for_dates = (
            copied_events_df[["event_date"]]
            if "event_date" in copied_events_df.columns
            else pd.DataFrame()
        )

        quick_range = st.selectbox(
            "📅 Time Period",
            [
                "Custom Range",
                "Today",
                "This Week",
                "Last Week",
                "This Month",
                "Last Month",
                "This Year",
                "Last Year",
            ],
        )

        # Use the original date range helper (returns Python date objects)
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        # Get valid aggregation levels based on date range
        available_aggregations = get_available_aggregations(start_date, end_date)

        # If current selection is not valid, fallback to widest available
        if (
            "period_label" not in st.session_state
            or st.session_state.period_label not in available_aggregations
        ):
            st.session_state.period_label = available_aggregations[-1]

        # Show selector (with safe default applied)
        period_label = st.selectbox(
            "⏰ Aggregation Level",
            available_aggregations,
            index=available_aggregations.index(st.session_state.period_label),
        )

        bg_color = st.color_picker("🎨 Chart Background", "#FFFFFF")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- APPLY FILTER ----------------
    # Convert date objects to datetimes for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # Filter events by selected range
    filtered_events = copied_events_df[
        (copied_events_df["event_date"] >= start_datetime)
        & (copied_events_df["event_date"] <= end_datetime)
    ].copy()

    # GAUGE - Store filtered events in session state
    st.session_state["filtered_events"] = filtered_events.copy()

    # Assign period AFTER filtering
    filtered_events = assign_period(filtered_events, "event_date", period_label)

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    text_color = get_text_color(bg_color)

    with col_chart:
        st.markdown(
            f'<div class="section-header">📈 {kpi_selection} Trend</div>',
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

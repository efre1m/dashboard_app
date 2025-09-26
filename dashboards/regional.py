import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.queries import get_facilities_for_user, get_facility_mapping_for_user
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    get_kpi_selection,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
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

    # ---------------- Facility Filter ----------------
    # Get facilities from database using queries.py
    db_facilities = get_facilities_for_user(user)
    facilities = [facility[0] for facility in db_facilities]  # Extract facility names

    # Create facility mapping for UID lookup (from database)
    facility_mapping = get_facility_mapping_for_user(user)

    # Multi-select facility selector in sidebar
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

    # Get the facility UIDs for selected facilities (from database mapping)
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

    # MAIN HEADING
    if selected_facilities == ["All Facilities"]:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Health Dashboard - {region_name}</div>',
            unsafe_allow_html=True,
        )
    elif len(selected_facilities) == 1:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Health Dashboard - {selected_facilities[0]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="main-header">🏥 Maternal Health Dashboard - Multiple Facilities ({len(selected_facilities)})</div>',
            unsafe_allow_html=True,
        )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # 🔒 Always show REGIONAL KPI values (ignore facility filters)
    all_facility_uids = list(facility_mapping.values())  # all facilities in this region

    # Use the REGION name for display (not country)
    display_name = region_name

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(
        user.get("id", user.get("username", "default_user"))
    )  # Prefer numeric ID, fallback to username

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

    # ---------------- APPLY FILTER (PRESERVING ORIGINAL LOGIC) ----------------
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

    # Filter enrollments by selected range (preserved for completeness)
    filtered_enrollments = enrollments_df.copy()
    if (
        not filtered_enrollments.empty
        and "enrollmentDate" in filtered_enrollments.columns
    ):
        filtered_enrollments = filtered_enrollments[
            (filtered_enrollments["enrollmentDate"] >= start_datetime)
            & (filtered_enrollments["enrollmentDate"] <= end_datetime)
        ]

    # Apply facility filter if selected
    if facility_uids:
        filtered_events = filtered_events[
            filtered_events["orgUnit"].isin(facility_uids)
        ]

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
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header">📈 {kpi_selection} - Facility Comparison</div>',
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
                f'<div class="section-header">📈 {kpi_selection} Trend</div>',
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

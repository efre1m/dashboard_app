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

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# ---------------- Cache Wrapper ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)


# ---------------- Initialize Session State ----------------
if "refresh_trigger" not in st.session_state:
    st.session_state.refresh_trigger = False

if "all_facilities_checkbox" not in st.session_state:
    st.session_state.all_facilities_checkbox = False

if "all_regions_checkbox" not in st.session_state:
    st.session_state.all_regions_checkbox = False

if "selected_facilities" not in st.session_state:
    st.session_state.selected_facilities = []

if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = []

if "expanded_regions" not in st.session_state:
    st.session_state.expanded_regions = {}


# Helper functions for facility and region selection
def handle_all_facilities_change(all_facility_names):
    """Handle when 'All Facilities' checkbox is toggled"""
    if st.session_state.all_facilities_checkbox:
        # Select all facilities
        st.session_state.selected_facilities = ["All Facilities"] + all_facility_names
        # Deselect all regions
        st.session_state.selected_regions = []
    else:
        # Deselect all facilities
        st.session_state.selected_facilities = []


def handle_region_change(region_name, facility_names):
    """Handle when 'Select all in region' checkbox is toggled"""
    if st.session_state[f"select_all_{region_name}"]:
        # Add all facilities in this region
        for facility_name in facility_names:
            if facility_name not in st.session_state.selected_facilities:
                st.session_state.selected_facilities.append(facility_name)
        # Remove "All Facilities" if individual facilities are selected
        if "All Facilities" in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove("All Facilities")
    else:
        # Remove all facilities from this region
        for facility_name in facility_names:
            if facility_name in st.session_state.selected_facilities:
                st.session_state.selected_facilities.remove(facility_name)


def handle_facility_change(facility_name):
    """Handle when individual facility checkbox is toggled"""
    if st.session_state[f"facility_{facility_name}"]:
        # Add this facility
        if facility_name not in st.session_state.selected_facilities:
            st.session_state.selected_facilities.append(facility_name)
        # Remove "All Facilities" if individual facilities are selected
        if "All Facilities" in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove("All Facilities")
    else:
        # Remove this facility
        if facility_name in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove(facility_name)


def handle_region_selection_change(region_name):
    """Handle when region checkbox is toggled for region comparison"""
    if st.session_state[f"region_{region_name}"]:
        # Add this region to selected regions
        if region_name not in st.session_state.selected_regions:
            st.session_state.selected_regions.append(region_name)
        # Clear facility selection when selecting regions
        st.session_state.selected_facilities = []
        st.session_state.all_facilities_checkbox = False
    else:
        # Remove this region
        if region_name in st.session_state.selected_regions:
            st.session_state.selected_regions.remove(region_name)


def handle_all_regions_change(all_region_names):
    """Handle when 'All Regions' checkbox is toggled"""
    if st.session_state.all_regions_checkbox:
        # Select all regions
        st.session_state.selected_regions = all_region_names.copy()
        # Clear facility selection
        st.session_state.selected_facilities = []
        st.session_state.all_facilities_checkbox = False
    else:
        # Deselect all regions
        st.session_state.selected_regions = []


# ---------------- Page Rendering ----------------
def render():
    st.set_page_config(
        page_title="National Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

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

    # Initialize session state for selections if not present
    if "current_facility_uids" not in st.session_state:
        st.session_state.current_facility_uids = list(facility_mapping.values())
    if "current_display_names" not in st.session_state:
        st.session_state.current_display_names = ["All Facilities"]
    if "current_comparison_mode" not in st.session_state:
        st.session_state.current_comparison_mode = "facility"

    # ---- Sidebar Filter Form ----
    with st.sidebar.form("facility_filter_form"):
        st.markdown(
            '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Facility Selection Mode</p>',
            unsafe_allow_html=True,
        )

        filter_mode = st.radio(
            "Select facilities by:",
            ["All Facilities", "By Region", "By Facility"],
            index=0,
        )

        selected_facilities, selected_regions = [], []

        if filter_mode == "By Region":
            st.markdown("**🌍 Select Regions**")
            all_regions_selected = st.checkbox("All Regions")
            if all_regions_selected:
                selected_regions = all_region_names.copy()
            else:
                for region in all_region_names:
                    if st.checkbox(region, key=f"region_{region}"):
                        selected_regions.append(region)

        elif filter_mode == "By Facility":
            st.markdown("**🏢 Select Facilities (grouped by region)**")
            for region_name, facilities in facilities_by_region.items():
                with st.expander(f"{region_name} ({len(facilities)})", expanded=False):
                    select_all_region = st.checkbox(
                        f"Select all in {region_name}", key=f"select_all_{region_name}"
                    )
                    if select_all_region:
                        for fac_name, _ in facilities:
                            selected_facilities.append(fac_name)
                    else:
                        for fac_name, _ in facilities:
                            if st.checkbox(fac_name, key=f"facility_{fac_name}"):
                                selected_facilities.append(fac_name)

        # Apply button
        submitted = st.form_submit_button("✅ Apply Filters")

    # ---- Apply the selection only if Apply button is clicked ----
    if submitted:
        if filter_mode == "All Facilities":
            facility_uids = list(facility_mapping.values())
            display_names = ["All Facilities"]
            comparison_mode = "facility"

        elif filter_mode == "By Region" and selected_regions:
            facility_uids, display_names = [], []
            for region in selected_regions:
                if region in facilities_by_region:
                    for fac_name, fac_uid in facilities_by_region[region]:
                        facility_uids.append(fac_uid)
            display_names = selected_regions
            comparison_mode = "region"

        elif filter_mode == "By Facility" and selected_facilities:
            facility_uids = [
                facility_mapping[f]
                for f in selected_facilities
                if f in facility_mapping
            ]
            display_names = selected_facilities
            comparison_mode = "facility"

        else:
            # fallback
            facility_uids = list(facility_mapping.values())
            display_names = ["All Facilities"]
            comparison_mode = "facility"

        # Save selections in session_state
        st.session_state.current_facility_uids = facility_uids
        st.session_state.current_display_names = display_names
        st.session_state.current_comparison_mode = comparison_mode

    # ---- Use session_state for the rest of dashboard ----
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

    # MAIN HEADING
    if (
        comparison_mode == "facility"
        and "All Facilities" in st.session_state.selected_facilities
    ):
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {country_name}</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "facility" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]}</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "facility":
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Facilities ({len(display_names)})</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]} Region</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Regions ({len(display_names)})</div>',
            unsafe_allow_html=True,
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

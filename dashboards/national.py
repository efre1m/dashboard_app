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


# Initialize session state keys at the very top
if "all_facilities_checkbox" not in st.session_state:
    st.session_state.all_facilities_checkbox = False


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
    # Get facilities grouped by region from database
    facilities_by_region = get_facilities_grouped_by_region(user)

    # Create facility mapping for UID lookup (from database)
    facility_mapping = get_facility_mapping_for_user(user)

    # Get all facility names for "All Facilities" selection
    all_facility_names = []
    for region_name, facilities in facilities_by_region.items():
        for facility_name, _ in facilities:
            all_facility_names.append(facility_name)

    # Get all region names
    all_region_names = list(facilities_by_region.keys())

    # Initialize session state for facility and region selection
    if "selected_facilities" not in st.session_state:
        st.session_state.selected_facilities = ["All Facilities"] + all_facility_names

    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = []

    if "expanded_regions" not in st.session_state:
        # Default all regions to collapsed (not expanded)
        st.session_state.expanded_regions = {
            region_name: False for region_name in facilities_by_region.keys()
        }

    # Facility and region selector in sidebar with expandable regions
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Select Facilities or Regions</p>',
        unsafe_allow_html=True,
    )

    # Create a container for the selector
    selector_container = st.sidebar.container()

    with selector_container:
        # "All Facilities" checkbox
        all_facilities_selected = st.checkbox(
            "All Facilities",
            value="All Facilities" in st.session_state.selected_facilities,
            key="all_facilities_checkbox",
            on_change=lambda: handle_all_facilities_change(all_facility_names),
        )

        # "All Regions" checkbox
        all_regions_selected = st.checkbox(
            "All Regions",
            value=len(st.session_state.selected_regions) == len(all_region_names),
            key="all_regions_checkbox",
            on_change=lambda: handle_all_regions_change(all_region_names),
        )

        # Individual region checkboxes for region comparison
        st.markdown(
            '<p style="color: white; font-weight: 600; margin: 15px 0 8px 0;">Select Regions for Comparison:</p>',
            unsafe_allow_html=True,
        )

        for region_name in all_region_names:
            region_selected = st.checkbox(
                region_name,
                value=region_name in st.session_state.selected_regions,
                key=f"region_{region_name}",
                on_change=lambda rn=region_name: handle_region_selection_change(rn),
            )

        # Region expanders for facility selection
        st.markdown(
            '<p style="color: white; font-weight: 600; margin: 15px 0 8px 0;">Select Individual Facilities:</p>',
            unsafe_allow_html=True,
        )

        for region_name, facilities in facilities_by_region.items():
            # Create expander for each region
            with st.expander(
                f"{region_name} ({len(facilities)})",
                expanded=st.session_state.expanded_regions[region_name],
            ):
                # Update expanded state when user expands
                st.session_state.expanded_regions[region_name] = True

                # Select all facilities in this region
                all_in_region_selected = all(
                    [
                        facility_name in st.session_state.selected_facilities
                        for facility_name, _ in facilities
                    ]
                )
                all_in_region = st.checkbox(
                    f"Select all in {region_name}",
                    value=all_in_region_selected,
                    key=f"select_all_{region_name}",
                    on_change=lambda r=region_name, f=[
                        fac[0] for fac in facilities
                    ]: handle_region_change(r, f),
                )

                # Individual facility checkboxes
                for facility_name, _ in facilities:
                    facility_selected = st.checkbox(
                        facility_name,
                        value=facility_name in st.session_state.selected_facilities,
                        key=f"facility_{facility_name}",
                        on_change=lambda fn=facility_name: handle_facility_change(fn),
                    )

    # Get the selected facilities and regions
    selected_facilities = [
        f for f in st.session_state.selected_facilities if f != "All Facilities"
    ]
    selected_regions = st.session_state.selected_regions

    # Calculate total facilities count
    total_facilities = sum(
        len(facilities) for facilities in facilities_by_region.values()
    )

    # Display selection count
    if "All Facilities" in st.session_state.selected_facilities:
        display_text = f"Selected: All Facilities ({total_facilities})"
    elif selected_facilities:
        display_text = (
            f"Selected: {len(selected_facilities)} / {total_facilities} Facilities"
        )
    elif selected_regions:
        if len(selected_regions) == len(all_region_names):
            display_text = f"Selected: All Regions ({len(selected_regions)})"
        else:
            display_text = (
                f"Selected: {len(selected_regions)} / {len(all_region_names)} Regions"
            )
    else:
        display_text = "No selection"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: 10px;'>{display_text}</p>",
        unsafe_allow_html=True,
    )

    # Get the facility UIDs for selected facilities or regions
    facility_uids = None
    display_names = None
    comparison_mode = None

    if selected_facilities:
        # Use selected individual facilities
        facility_uids = [
            facility_mapping[facility]
            for facility in selected_facilities
            if facility in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    elif selected_regions:
        # Use all facilities in selected regions
        facility_uids = []
        for region_name in selected_regions:
            if region_name in facilities_by_region:
                for facility_name, facility_uid in facilities_by_region[region_name]:
                    facility_uids.append(facility_uid)
        display_names = selected_regions
        comparison_mode = "region"
    else:
        # Default to all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

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

# dashboards/regional.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from newborns_dashboard.newborn_dashboard import render_newborn_dashboard
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
    render_kpi_tab_navigation,  # ADD THIS IMPORT
)
from utils.kpi_utils import clear_cache, compute_kpis
from utils.kpi_lbw import compute_lbw_kpi
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


def calculate_maternal_indicators(maternal_events_df, facility_uids):
    """Calculate maternal indicators using appropriate KPI functions"""
    if maternal_events_df.empty:
        return {
            "total_deliveries": 0,
            "maternal_deaths": 0,
            "maternal_death_rate": 0.0,
            "live_births": 0,
            "stillbirths": 0,
            "total_births": 0,
            "stillbirth_rate": 0.0,
            "low_birth_weight_rate": 0.0,
            "low_birth_weight_count": 0,
        }

    # Use compute_kpis for basic maternal indicators
    kpi_data = compute_kpis(maternal_events_df, facility_uids)

    total_deliveries = kpi_data.get("total_deliveries", 0)
    maternal_deaths = kpi_data.get("maternal_deaths", 0)
    live_births = kpi_data.get("live_births", 0)
    stillbirths = kpi_data.get("stillbirths", 0)
    total_births = kpi_data.get("total_births", 0)

    # Use compute_lbw_kpi specifically for low birth weight data
    lbw_data = compute_lbw_kpi(maternal_events_df, facility_uids)
    low_birth_weight_count = lbw_data.get("lbw_count", 0)
    total_weighed = lbw_data.get("total_weighed", total_births)
    low_birth_weight_rate = lbw_data.get("lbw_rate", 0.0)

    # Calculate rates for indicators not provided by specialized functions
    maternal_death_rate = (
        (maternal_deaths / live_births * 100000) if live_births > 0 else 0.0
    )
    stillbirth_rate = (stillbirths / total_births * 1000) if total_births > 0 else 0.0

    # If lbw_rate is 0 but we have data, calculate it manually
    if low_birth_weight_rate == 0 and total_weighed > 0:
        low_birth_weight_rate = low_birth_weight_count / total_weighed * 100

    return {
        "total_deliveries": total_deliveries,
        "maternal_deaths": maternal_deaths,
        "maternal_death_rate": round(maternal_death_rate, 2),
        "live_births": live_births,
        "stillbirths": stillbirths,
        "total_births": total_births,
        "stillbirth_rate": round(stillbirth_rate, 2),
        "low_birth_weight_rate": round(low_birth_weight_rate, 2),
        "low_birth_weight_count": low_birth_weight_count,
    }


def calculate_newborn_indicators(newborn_events_df, facility_uids):
    """Calculate newborn indicators"""
    if newborn_events_df.empty:
        return {"total_admitted": 0, "nmr": 0.0}

    # For newborn admitted count, we need to count from filtered TEI data separately
    total_admitted = 0  # Placeholder - will be set from filtered TEI count
    nmr = 0.0  # Placeholder - will need specific NMR calculation

    return {"total_admitted": total_admitted, "nmr": nmr}


def get_location_display_name(selected_facilities, region_name):
    """Get the display name for location based on selection - FOLLOWING NATIONAL.PY PATTERN"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        # Join multiple facilities with comma - EXACTLY LIKE NATIONAL.PY
        return ", ".join(selected_facilities), "Facilities"


def render_summary_dashboard(
    user, region_name, facility_mapping, selected_facilities, facility_uids
):
    """Render Summary Dashboard with both maternal and newborn overview tables"""

    # Get location display name - FOLLOWING NATIONAL.PY PATTERN
    location_name, location_type = get_location_display_name(
        selected_facilities, region_name
    )

    st.markdown(
        f'<div class="main-header">📊 Summary Dashboard - {location_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Comprehensive overview of maternal and newborn health indicators**")

    # Get programs for UID mapping
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    # Fetch both maternal and newborn data
    with st.spinner("Fetching summary data for both programs..."):
        try:
            # Fetch maternal data
            maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
            maternal_dfs = (
                fetch_cached_data(user, maternal_program_uid)
                if maternal_program_uid
                else {}
            )

            # Fetch newborn data
            newborn_program_uid = program_uid_map.get("Newborn Care Form")
            newborn_dfs = (
                fetch_cached_data(user, newborn_program_uid)
                if newborn_program_uid
                else {}
            )

            update_last_sync_time()

        except Exception as e:
            st.error(f"⚠️ Error fetching summary data: {e}")
            return

    # Extract dataframes
    maternal_tei_df = maternal_dfs.get("tei", pd.DataFrame())
    maternal_events_df = maternal_dfs.get("events", pd.DataFrame())
    newborn_tei_df = newborn_dfs.get("tei", pd.DataFrame())
    newborn_events_df = newborn_dfs.get("events", pd.DataFrame())

    # Normalize dates for start date calculation
    newborn_enrollments_df = newborn_dfs.get("enrollments", pd.DataFrame())
    newborn_enrollments_df = normalize_enrollment_dates(newborn_enrollments_df)

    maternal_enrollments_df = maternal_dfs.get("enrollments", pd.DataFrame())
    maternal_enrollments_df = normalize_enrollment_dates(maternal_enrollments_df)

    # Calculate indicators
    maternal_indicators = calculate_maternal_indicators(
        maternal_events_df, facility_uids
    )
    newborn_indicators = calculate_newborn_indicators(newborn_events_df, facility_uids)

    # Get filtered TEI counts based on facility selection
    maternal_tei_count = count_unique_teis_filtered(
        maternal_tei_df, facility_uids, "tei_orgUnit"
    )
    newborn_tei_count = count_unique_teis_filtered(
        newborn_tei_df, facility_uids, "tei_orgUnit"
    )

    # Update newborn indicators with correct TEI count
    newborn_indicators["total_admitted"] = newborn_tei_count

    # Get earliest dates from enrollments
    newborn_start_date = get_earliest_date(newborn_enrollments_df, "enrollmentDate")
    maternal_start_date = get_earliest_date(maternal_enrollments_df, "enrollmentDate")

    # Apply professional table styling - EXACTLY LIKE NATIONAL.PY
    st.markdown(
        """
    <style>
    .professional-table {
        font-family: 'Arial', sans-serif;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .professional-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .professional-table th {
        background: linear-gradient(135deg, #1f77b4, #1668a1);
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        border: none;
    }
    .professional-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 14px;
        background-color: white;
    }
    .professional-table tr:last-child td {
        border-bottom: none;
    }
    .professional-table tr:hover td {
        background-color: #f8f9fa;
    }
    .newborn-header {
        background: linear-gradient(135deg, #1f77b4, #1668a1) !important;
    }
    .maternal-header {
        background: linear-gradient(135deg, #2ca02c, #228b22) !important;
    }
    .indicator-cell {
        font-weight: 500;
        color: #333;
    }
    .value-cell {
        font-weight: 600;
        color: #1a1a1a;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create overview tables with professional styling
    st.markdown("---")

    # Newborn Overview Table - Professional Styling - EXACTLY LIKE NATIONAL.PY
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 👶 Newborn Care Overview")

    with col2:
        # Create newborn data for download
        newborn_data = {
            "Start Date": [newborn_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Newborns": [newborn_tei_count],
            "NMR": [f"{newborn_indicators['nmr']}%"],
            "Stillbirth Rate": [
                f"{maternal_indicators['stillbirth_rate']} per 1000 births"
            ],
            "Live Births": [maternal_indicators["live_births"]],
            "Stillbirths": [maternal_indicators["stillbirths"]],
            "Total Births": [maternal_indicators["total_births"]],
            "Low Birth Weight Rate": [
                f"{maternal_indicators['low_birth_weight_rate']}%"
            ],
        }

        newborn_df = pd.DataFrame(newborn_data)
        newborn_csv = newborn_df.to_csv(index=False)

        st.download_button(
            "📥 Download Newborn Data",
            data=newborn_csv,
            file_name=f"newborn_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Create newborn table data
    newborn_table_data = {
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Newborns",
            "NMR",
            "Stillbirth Rate",
            "Live Births",
            "Stillbirths",
            "Total Births",
            "Low Birth Weight Rate (<2500g)",
        ],
        "Value": [
            newborn_start_date,
            location_name,
            f"{newborn_tei_count:,}",
            f"{newborn_indicators['nmr']}%",
            f"{maternal_indicators['stillbirth_rate']} per 1000 births",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['total_births']:,}",
            f"{maternal_indicators['low_birth_weight_rate']}%",
        ],
    }

    newborn_table_df = pd.DataFrame(newborn_table_data)

    # Display styled newborn table
    st.markdown('<div class="professional-table">', unsafe_allow_html=True)
    st.table(newborn_table_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # Maternal Overview Table - Professional Styling - EXACTLY LIKE NATIONAL.PY
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 🤰 Maternal Care Overview")

    with col2:
        # Create maternal data for download
        maternal_data = {
            "Start Date": [maternal_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Mothers": [maternal_tei_count],
            "Total Deliveries": [maternal_indicators["total_deliveries"]],
            "Maternal Death Rate": [
                f"{maternal_indicators['maternal_death_rate']} per 100,000 births"
            ],
        }

        maternal_df = pd.DataFrame(maternal_data)
        maternal_csv = maternal_df.to_csv(index=False)

        st.download_button(
            "📥 Download Maternal Data",
            data=maternal_csv,
            file_name=f"maternal_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Create maternal table data
    maternal_table_data = {
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Mothers",
            "Total Deliveries",
            "Maternal Death Rate",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_indicators['total_deliveries']:,}",
            f"{maternal_indicators['maternal_death_rate']} per 100,000 births",
        ],
    }

    maternal_table_df = pd.DataFrame(maternal_table_data)

    # Display styled maternal table
    st.markdown('<div class="professional-table">', unsafe_allow_html=True)
    st.table(maternal_table_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary statistics cards - EXACTLY LIKE NATIONAL.PY
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Mothers", f"{maternal_tei_count:,}", help="Unique mothers admitted"
        )

    with col2:
        st.metric(
            "Total Newborns", f"{newborn_tei_count:,}", help="Unique newborns admitted"
        )

    with col3:
        st.metric(
            "Total Deliveries",
            f"{maternal_indicators['total_deliveries']:,}",
            help="Total number of deliveries",
        )

    with col4:
        st.metric("Coverage Start", maternal_start_date, help="Earliest data record")


def render_maternal_dashboard(
    user,
    program_uid,
    region_name,
    selected_facilities,
    facility_uids,
    view_mode,
    facility_mapping,
    facility_names,
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

    # MAIN HEADING for Maternal program - FOLLOWING NATIONAL.PY PATTERN FOR FACILITY DISPLAY
    if selected_facilities == ["All Facilities"]:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {region_name}</div>',
            unsafe_allow_html=True,
        )
    elif len(selected_facilities) == 1:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {selected_facilities[0]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Display facility names separated by comma - EXACTLY LIKE NATIONAL.PY
        facilities_display = ", ".join(selected_facilities)
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {facilities_display}</div>',
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
        filters = render_simple_filter_controls(
            copied_events_df, container=col_ctrl, context="regional_maternal"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(copied_events_df, filters, facility_uids)

    # Store for gauge charts
    st.session_state["filtered_events"] = filtered_events.copy()

    # Get variables from filters for later use
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
        # NEW: Use KPI tab navigation instead of filters["kpi_selection"]
        selected_kpi = render_kpi_tab_navigation()

        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header">📈 {selected_kpi} - Facility Comparison - Maternal Inpatient Data</div>',  # Use selected_kpi instead of kpi_selection
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use common comparison chart function
            render_comparison_chart(
                kpi_selection=selected_kpi,  # Use selected_kpi here
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
                f'<div class="section-header">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',  # Use selected_kpi here
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Use common trend chart function
            render_trend_chart_section(
                selected_kpi,  # Use selected_kpi here
                filtered_events,
                facility_uids,
                facility_names,
                bg_color,
                text_color,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Use common additional analytics function
        render_additional_analytics(
            selected_kpi,  # Use selected_kpi here
            filtered_events,
            facility_uids,
            bg_color,
            text_color,
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

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        clear_cache()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # ---------------- Get Facilities Data ----------------
    db_facilities = get_facilities_for_user(user)
    facilities = [facility[0] for facility in db_facilities]
    facility_mapping = get_facility_mapping_for_user(user)

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
    else:
        # All facilities selected - use all facility UIDs in the region
        facility_uids = list(facility_mapping.values())
        facility_names = ["All Facilities"]

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if selected_facilities != ["All Facilities"] and len(selected_facilities) > 1:
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Facility Comparison"],
            index=0,
            help="Compare trends across multiple facilities",
        )

    # Get programs for UID mapping
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    # CREATE PROFESSIONAL TABS IN MAIN AREA - ADDED OVERVIEW TAB
    tab1, tab2, tab3 = st.tabs(
        [
            "🤰 **Maternal Inpatient Data**",
            "👶 **Newborn Care Form**",
            "📊 **Summary Dashboard**",
        ]
    )

    with tab1:
        maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
        if maternal_program_uid:
            render_maternal_dashboard(
                user,
                maternal_program_uid,
                region_name,
                selected_facilities,
                facility_uids,
                view_mode,
                facility_mapping,
                facility_names,
            )
        else:
            st.error("Maternal Inpatient Data program not found")

    with tab2:
        newborn_program_uid = program_uid_map.get("Newborn Care Form")
        if newborn_program_uid:
            render_newborn_dashboard(
                user,
                newborn_program_uid,
                region_name=region_name,
                selected_facilities=selected_facilities,
                facility_uids=facility_uids,
                facility_mapping=facility_mapping,
                facility_names=facility_names,
                view_mode=view_mode,
                country_name=None,
                facilities_by_region=None,
            )
        else:
            st.error("Newborn Care Form program not found")

    with tab3:
        # NEW: Summary Dashboard Tab
        render_summary_dashboard(
            user,
            region_name,
            facility_mapping,
            selected_facilities,
            facility_uids,
        )

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
from newborns_dashboard.facility_newborn import render_newborn_dashboard
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    render_trend_chart_section,
    render_additional_analytics,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
    render_kpi_tab_navigation,  # ADD THIS IMPORT
)
from utils.kpi_utils import clear_cache, compute_kpis  # ADD compute_kpis IMPORT
from utils.kpi_lbw import compute_lbw_kpi  # ADD THIS IMPORT
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)

# from utils.odk_dashboard import display_odk_dashboard

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


def count_unique_teis_filtered(tei_df, facility_uids, org_unit_column="tei_orgUnit"):
    """Count unique TEIs from tei_df - SIMPLE VERSION THAT WORKS"""
    if tei_df.empty:
        # If TEI dataframe is empty, try to get count from events data
        return 0

    # JUST COUNT WHATEVER TEIs ARE IN THE DATAFRAME - NO FILTERING
    # The data should already be filtered for this facility when fetched

    # Try common TEI identifier columns
    if "tei" in tei_df.columns:
        return tei_df["tei"].nunique()
    elif "tei_id" in tei_df.columns:
        return tei_df["tei_id"].nunique()
    elif "trackedEntityInstance" in tei_df.columns:
        return tei_df["trackedEntityInstance"].nunique()
    elif "tracked_entity_instance" in tei_df.columns:
        return tei_df["tracked_entity_instance"].nunique()
    else:
        # If no TEI column found, show available columns and count rows
        print(f"DEBUG - Available columns: {tei_df.columns.tolist()}")
        return len(tei_df)  # Fallback: count all rows


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
        return {"total_admitted": 0, "nmr": "N/A"}

    # For newborn admitted count, we need to count from filtered TEI data separately
    total_admitted = 0  # Placeholder - will be set from filtered TEI count
    nmr = "N/A"  # Placeholder - will need specific NMR calculation

    return {"total_admitted": total_admitted, "nmr": nmr}


def get_location_display_name(facility_name):
    """Get the display name for location - SIMPLIFIED FOR FACILITY LEVEL"""
    return facility_name, "Facility"


def render_summary_dashboard(user, facility_name, facility_uid):
    """Render Summary Dashboard with both maternal and newborn overview tables"""

    # Get location display name - SIMPLIFIED FOR FACILITY LEVEL
    location_name, location_type = get_location_display_name(facility_name)

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

    # Filter data to only show this facility's data
    if facility_uid and not maternal_events_df.empty:
        maternal_events_df = maternal_events_df[
            maternal_events_df["orgUnit"] == facility_uid
        ]
    if facility_uid and not newborn_events_df.empty:
        newborn_events_df = newborn_events_df[
            newborn_events_df["orgUnit"] == facility_uid
        ]

    # Normalize dates for start date calculation
    newborn_enrollments_df = newborn_dfs.get("enrollments", pd.DataFrame())
    newborn_enrollments_df = normalize_enrollment_dates(newborn_enrollments_df)

    maternal_enrollments_df = maternal_dfs.get("enrollments", pd.DataFrame())
    maternal_enrollments_df = normalize_enrollment_dates(maternal_enrollments_df)

    # Filter enrollments by facility
    if facility_uid and not newborn_enrollments_df.empty:
        newborn_enrollments_df = newborn_enrollments_df[
            newborn_enrollments_df["orgUnit"] == facility_uid
        ]
    if facility_uid and not maternal_enrollments_df.empty:
        maternal_enrollments_df = maternal_enrollments_df[
            maternal_enrollments_df["orgUnit"] == facility_uid
        ]

    # Calculate indicators
    maternal_indicators = calculate_maternal_indicators(
        maternal_events_df, facility_uid
    )
    newborn_indicators = calculate_newborn_indicators(newborn_events_df, facility_uid)

    # Get filtered TEI counts based on facility selection
    maternal_tei_count = count_unique_teis_filtered(
        maternal_tei_df, facility_uid, "tei_orgUnit"
    )
    newborn_tei_count = count_unique_teis_filtered(
        newborn_tei_df, facility_uid, "tei_orgUnit"
    )

    # Update newborn indicators with correct TEI count
    newborn_indicators["total_admitted"] = newborn_tei_count

    # Get earliest dates from enrollments
    newborn_start_date = get_earliest_date(newborn_enrollments_df, "enrollmentDate")
    maternal_start_date = get_earliest_date(maternal_enrollments_df, "enrollmentDate")

    # Apply professional table styling - MATCHING KPI UTILS STYLING
    st.markdown(
        """
    <style>
    .summary-table-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }

    .summary-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
    }

    .summary-table thead tr {
        background: linear-gradient(135deg, #1f77b4, #1668a1);
    }

    .summary-table th {
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        border: none;
    }

    .summary-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 14px;
        background-color: white;
    }

    .summary-table tbody tr:last-child td {
        border-bottom: none;
    }

    .summary-table tbody tr:hover td {
        background-color: #f8f9fa;
    }

    /* Specific styling for different table types */
    .newborn-table thead tr {
        background: linear-gradient(135deg, #1f77b4, #1668a1) !important;
    }

    .maternal-table thead tr {
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

    /* Number column styling */
    .summary-table td:first-child {
        font-weight: 600;
        color: #666;
        text-align: center;
    }

    .summary-table th:first-child {
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create overview tables with professional styling
    st.markdown("---")

    # Newborn Overview Table - Professional Styling
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 👶 Newborn Care Overview")

    with col2:
        # Create newborn data for download
        newborn_data = {
            "Start Date": [newborn_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Newborns": [newborn_tei_count],
            "NMR": [f"{newborn_indicators['nmr']}"],
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

    # Create newborn table data WITH NUMBERING
    newborn_table_data = {
        "No": list(range(1, 10)),
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
            f"{newborn_indicators['nmr']}",
            f"{maternal_indicators['stillbirth_rate']} per 1000 births",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['total_births']:,}",
            f"{maternal_indicators['low_birth_weight_rate']}%",
        ],
    }

    newborn_table_df = pd.DataFrame(newborn_table_data)

    # Display styled newborn table - USING KPI UTILS STYLING
    st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
    st.markdown(
        newborn_table_df.style.set_table_attributes(
            'class="summary-table newborn-table"'
        )
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [("color", "white"), ("font-weight", "600")],
                },
                {
                    "selector": "tbody td",
                    "props": [("border-bottom", "1px solid #f0f0f0")],
                },
                {
                    "selector": "tbody tr:last-child td",
                    "props": [("border-bottom", "none")],
                },
            ]
        )
        .to_html(),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Maternal Overview Table - Professional Styling
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

    # Create maternal table data WITH NUMBERING
    maternal_table_data = {
        "No": list(range(1, 6)),
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

    # Display styled maternal table - USING KPI UTILS STYLING
    st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
    st.markdown(
        maternal_table_df.style.set_table_attributes(
            'class="summary-table maternal-table"'
        )
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [("color", "white"), ("font-weight", "600")],
                },
                {
                    "selector": "tbody td",
                    "props": [("border-bottom", "1px solid #f0f0f0")],
                },
                {
                    "selector": "tbody tr:last-child td",
                    "props": [("border-bottom", "none")],
                },
            ]
        )
        .to_html(),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary statistics cards - PROFESSIONAL STYLING
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")

    # Add professional CSS styling for the metrics
    st.markdown(
        """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.2);
        text-align: center;
        transition: transform 0.3s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .metric-help {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 5px;
        line-height: 1.3;
    }
    .critical-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%) !important;
    }
    .success-metric {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
    }
    .info-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    }
    .warning-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create the professional metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card info-metric">
            <div class="metric-label">👩 Total Mothers</div>
            <div class="metric-value">{maternal_tei_count:,}</div>
            <div class="metric-help">Unique mothers admitted</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card success-metric">
            <div class="metric-label">👶 Total Newborns</div>
            <div class="metric-value">{newborn_tei_count:,}</div>
            <div class="metric-help">Unique newborns admitted</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card critical-metric">
            <div class="metric-label">⚠️ Maternal Deaths</div>
            <div class="metric-value">{maternal_indicators['maternal_deaths']:,}</div>
            <div class="metric-help">Maternal mortality cases</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card warning-metric">
            <div class="metric-label">📅 Coverage Start</div>
            <div class="metric-value">{maternal_start_date}</div>
            <div class="metric-help">Earliest data record</div>
        </div>
        """,
            unsafe_allow_html=True,
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
        filters = render_simple_filter_controls(
            copied_events_df, container=col_ctrl, context="facility_maternal"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters with single facility UID
    filtered_events = apply_simple_filters(copied_events_df, filters, facility_uid)

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

        st.markdown(
            f'<div class="section-header">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',  # Use selected_kpi instead of kpi_selection
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Use common trend chart function with single facility
        render_trend_chart_section(
            selected_kpi,  # Use selected_kpi here
            filtered_events,
            facility_uid,  # Single facility UID
            facility_name,  # Single facility name
            bg_color,
            text_color,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Use common additional analytics function
        render_additional_analytics(
            selected_kpi,  # Use selected_kpi here
            filtered_events,
            facility_uid,
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

    # CREATE PROFESSIONAL TABS IN MAIN AREA - ADDED SUMMARY DASHBOARD TAB
    tab1, tab2, tab3 = st.tabs(
        [
            "🤰 **Maternal Inpatient Data**",
            "👶 **Newborn Inpatient Data**",
            "📊 **Summary Dashboard**",
        ]
    )

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
        newborn_program_uid = program_uid_map.get("Newborn Care Form")
        if newborn_program_uid:
            render_newborn_dashboard(
                user=user,
                program_uid=newborn_program_uid,
                facility_name=facility_name,
                facility_uid=facility_uid,
            )
        else:
            st.error("Newborn Care Form program not found")

    with tab3:
        # NEW: Summary Dashboard Tab
        render_summary_dashboard(user, facility_name, facility_uid)
    # with tab4:
    # NEW: ODK Forms Dashboard Tab
    # display_odk_dashboard(user)

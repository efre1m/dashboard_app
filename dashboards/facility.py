# dashboards/facility.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from utils.kpi_utils import clear_cache
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_for_user,
    get_facility_mapping_for_user,
)
from newborns_dashboard.facility_newborn import render_newborn_dashboard
from utils.dash_co import (
    normalize_patient_dates,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
    apply_patient_filters,
    render_patient_filter_controls,
    render_kpi_tab_navigation,
    KPI_OPTIONS,
)
from utils.kpi_utils import compute_kpis
from newborns_dashboard.kpi_nmr import compute_nmr_kpi

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# Performance optimization: Pre-load essential data with user-specific caching
@st.cache_data(ttl=3600, show_spinner=False)
def get_static_data_facility(user):
    """Cache static data for facility dashboard - user-specific"""
    user_identifier = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"

    facilities = get_facilities_for_user(user)
    facility_mapping = get_facility_mapping_for_user(user)
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    return {
        "facilities": facilities,
        "facility_mapping": facility_mapping,
        "program_uid_map": program_uid_map,
        "user_identifier": user_identifier,
    }


# Optimized shared cache for facility dashboard
@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=5)
def fetch_shared_program_data_facility(user, program_uid):
    """Optimized shared cache for program data - user-specific for facility"""
    if not program_uid:
        return None
    try:
        return fetch_program_data_for_user(user, program_uid)
    except Exception as e:
        logging.error(f"Error fetching data for program {program_uid}: {e}")
        return None


def get_shared_program_data_facility(user, program_uid_map, show_spinner=True):
    """Smart data loading with 30-minute auto-refresh for facility dashboard"""
    maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
    newborn_program_uid = program_uid_map.get("Newborn Care Form")

    # Create user-specific session state keys for facility
    user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    shared_loaded_key = f"shared_data_loaded_facility_{user_key}"
    shared_maternal_key = f"shared_maternal_data_facility_{user_key}"
    shared_newborn_key = f"shared_newborn_data_facility_{user_key}"
    shared_timestamp_key = f"shared_data_timestamp_facility_{user_key}"

    current_time = time.time()

    # Check if cache has expired (30 minutes = 1800 seconds)
    cache_expired = False
    if shared_timestamp_key in st.session_state:
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        cache_expired = time_elapsed > 1800
        if cache_expired:
            logging.info(
                f"Facility cache expired after {time_elapsed:.0f} seconds, fetching fresh data"
            )

    # Check if user changed (force fresh data)
    user_changed = st.session_state.get("user_changed_facility", False)

    # Determine if we need fresh data
    need_fresh_data = (
        not st.session_state.get(shared_loaded_key, False)
        or cache_expired
        or user_changed
        or st.session_state.get("refresh_trigger_facility", False)
    )

    if need_fresh_data:
        logging.info(
            "Fetching fresh facility data (cache expired, user changed, or manual refresh)"
        )

        # Clear existing cache
        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None

        # Load fresh data in parallel
        spinner_text = "Loading dashboard data..." if show_spinner else None

        def load_data():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                maternal_future = (
                    executor.submit(
                        fetch_shared_program_data_facility, user, maternal_program_uid
                    )
                    if maternal_program_uid
                    else None
                )
                newborn_future = (
                    executor.submit(
                        fetch_shared_program_data_facility, user, newborn_program_uid
                    )
                    if newborn_program_uid
                    else None
                )

                try:
                    if maternal_future:
                        st.session_state[shared_maternal_key] = maternal_future.result(
                            timeout=120
                        )
                    if newborn_future:
                        st.session_state[shared_newborn_key] = newborn_future.result(
                            timeout=120
                        )

                    st.session_state[shared_loaded_key] = True
                    st.session_state[shared_timestamp_key] = current_time

                    # Store in main cached_shared_data for easy access
                    st.session_state.cached_shared_data_facility = {
                        "maternal": st.session_state[shared_maternal_key],
                        "newborn": st.session_state[shared_newborn_key],
                    }

                    # Log fresh data stats
                    maternal_patient_count = (
                        len(
                            st.session_state[shared_maternal_key].get(
                                "patients", pd.DataFrame()
                            )
                        )
                        if st.session_state[shared_maternal_key]
                        else 0
                    )
                    newborn_patient_count = (
                        len(
                            st.session_state[shared_newborn_key].get(
                                "patients", pd.DataFrame()
                            )
                        )
                        if st.session_state[shared_newborn_key]
                        else 0
                    )
                    logging.info(
                        f"Fresh facility data: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients"
                    )

                    # Reset refresh trigger and user changed flags
                    st.session_state.refresh_trigger_facility = False
                    st.session_state.user_changed_facility = False

                except concurrent.futures.TimeoutError:
                    logging.error("Facility data loading timeout")
                    return False
            return True

        # Show spinner only if requested
        if show_spinner:
            with st.spinner(spinner_text):
                success = load_data()
                if not success:
                    st.error("Data loading timeout. Please try refreshing.")
        else:
            success = load_data()

    else:
        # Use cached data
        maternal_patient_count = (
            len(st.session_state[shared_maternal_key].get("patients", pd.DataFrame()))
            if st.session_state[shared_maternal_key]
            else 0
        )
        newborn_patient_count = (
            len(st.session_state[shared_newborn_key].get("patients", pd.DataFrame()))
            if st.session_state[shared_newborn_key]
            else 0
        )
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        logging.info(
            f"Using cached facility data: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients ({time_elapsed:.0f}s old)"
        )

    return {
        "maternal": st.session_state[shared_maternal_key],
        "newborn": st.session_state[shared_newborn_key],
    }


def clear_shared_cache_facility(user=None):
    """Clear shared data cache for facility dashboard"""
    if user:
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        shared_loaded_key = f"shared_data_loaded_facility_{user_key}"
        shared_maternal_key = f"shared_maternal_data_facility_{user_key}"
        shared_newborn_key = f"shared_newborn_data_facility_{user_key}"
        shared_timestamp_key = f"shared_data_timestamp_facility_{user_key}"

        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None
        if shared_timestamp_key in st.session_state:
            del st.session_state[shared_timestamp_key]

        logging.info("Cleared facility user-specific cache")
    else:
        # Clear all facility user caches
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith("shared_data_loaded_facility_")
            or key.startswith("shared_maternal_data_facility_")
            or key.startswith("shared_newborn_data_facility_")
            or key.startswith("shared_data_timestamp_facility_")
        ]
        for key in keys_to_clear:
            del st.session_state[key]
        logging.info("Cleared ALL facility shared caches")

    clear_cache()


def initialize_session_state_facility():
    """Optimized session state initialization for facility dashboard"""
    session_vars = {
        "refresh_trigger_facility": False,
        "current_facility_uids": [],
        "current_display_names": [],
        "current_comparison_mode": "facility",
        "filtered_patients": pd.DataFrame(),
        "selection_applied": True,
        "cached_patients_data": None,
        "cached_enrollments_data": None,
        "cached_tei_data": None,
        "last_applied_selection": None,
        "kpi_cache": {},
        "selected_program_uid": None,
        "selected_program_name": "Maternal Inpatient Data",
        "static_data_loaded_facility": False,
        "facility_filter_applied": False,
        "current_user_identifier_facility": None,
        "user_changed_facility": False,
        "last_computed_kpis": None,
        "last_computed_facilities": None,
        "last_computed_timestamp": None,
        "last_computed_newborn_kpis": None,
        "last_computed_newborn_timestamp": None,
        "summary_kpi_cache_facility": {},
        "active_tab": "maternal",
        "data_initialized_facility": False,
        "tab_initialized": {
            "maternal": False,
            "newborn": False,
            "summary": False,
        },
        "tab_data_loaded": {
            "maternal": True,
            "newborn": True,
            "summary": False,
        },
        "tab_loading": {
            "summary": False,
        },
        "facilities_facility": [],
        "facility_mapping_facility": {},
        "program_uid_map_facility": {},
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Check if user has changed
    current_user = st.session_state.get("user", {})
    new_user_identifier = f"{current_user.get('username', 'unknown')}_{current_user.get('role', 'unknown')}"

    if st.session_state.current_user_identifier_facility != new_user_identifier:
        st.session_state.user_changed_facility = True
        st.session_state.current_user_identifier_facility = new_user_identifier
        st.session_state.static_data_loaded_facility = False
        st.session_state.selection_applied = True
        st.session_state.facility_filter_applied = False
        st.session_state.last_computed_kpis = None
        st.session_state.last_computed_newborn_kpis = None
        st.session_state.summary_kpi_cache_facility = {}
        st.session_state.data_initialized_facility = False
        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
        st.session_state.tab_data_loaded["maternal"] = True
        st.session_state.tab_data_loaded["newborn"] = True
        st.session_state.tab_data_loaded["summary"] = False
        st.session_state.tab_loading["summary"] = False
    else:
        st.session_state.user_changed_facility = False


# Initialize session state at the very beginning
initialize_session_state_facility()


def count_unique_patients(patient_df, facility_uid, org_unit_column="orgUnit"):
    """Count unique patients from patient-level dataframe using UID"""
    if patient_df.empty:
        return 0

    # Filter by facility if specified
    if facility_uid and org_unit_column in patient_df.columns:
        filtered_patients = patient_df[patient_df[org_unit_column] == facility_uid]
    else:
        filtered_patients = patient_df

    # Count unique TEI IDs
    if "tei_id" in filtered_patients.columns:
        count = filtered_patients["tei_id"].nunique()
        return count
    else:
        return 0


def get_earliest_date(df, date_column):
    """Optimized date extraction"""
    if df.empty or date_column not in df.columns:
        return "N/A"

    try:
        earliest_date = df[date_column].min()
        return (
            earliest_date.strftime("%Y-%m-%d") if not pd.isna(earliest_date) else "N/A"
        )
    except Exception:
        return "N/A"


def get_location_display_name_facility(facility_name):
    """Get location display name for facility level"""
    return facility_name, "Facility"


# NEW: Maintenance message function for newborn dashboard
def render_newborn_maintenance_message():
    """Display maintenance message for newborn dashboard"""
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #fff8e1, #ffecb3);
         border-radius: 12px; border: 2px solid #ffb300; margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔧</div>
        <h2 style="color: #e65100; margin-bottom: 1rem;">Newborn Dashboard Under Maintenance</h2>
        <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            The newborn dashboard is currently undergoing maintenance and will be available soon.
            <br>In the meantime, you can view maternal health indicators in the other tabs.
        </p>
        <div style="display: inline-block; padding: 10px 20px; background: #ff9800; color: white;
             border-radius: 25px; font-weight: bold;">
            Maintenance in Progress
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_summary_dashboard_facility(user, facility_name, facility_uid, shared_data):
    """Optimized Summary Dashboard for facility - SIMPLIFIED with only 3 key indicators"""

    # Only run if this is the active tab
    if st.session_state.active_tab != "summary":
        return

    # Check if user has clicked "View Data" button for this tab
    if not st.session_state.tab_data_loaded["summary"]:
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
             border-radius: 12px; border: 2px dashed #dee2e6; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #495057; margin-bottom: 1rem;">Summary Dashboard</h2>
            <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                Get overview of key maternal health indicators for this facility
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "Load Summary Data",
                use_container_width=True,
                type="primary",
                key="load_summary_data_facility",
            ):
                st.session_state.tab_loading["summary"] = True
                st.session_state.tab_data_loaded["summary"] = True
                st.rerun()
        return

    # Show loading indicator if data is being processed
    if st.session_state.tab_loading["summary"]:
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
             border-radius: 12px; border: 2px solid #dee2e6; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #495057; margin-bottom: 1rem;">Loading Summary Dashboard...</h2>
            <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                Please wait while we process the data. This may take 1-2 minutes.
            </p>
            <div style="display: inline-block; padding: 10px 20px; background: #007bff; color: white; 
                 border-radius: 25px; font-weight: bold;">
                Processing Data...
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.session_state.tab_loading["summary"] = False
        st.rerun()

    logging.info("Facility summary dashboard rendering - SIMPLIFIED VERSION")

    # Get location display name
    location_name, location_type = get_location_display_name_facility(facility_name)

    # Use only maternal data (newborn data is under maintenance)
    maternal_data = shared_data["maternal"]

    if not maternal_data:
        st.error("No maternal data available for summary dashboard")
        return

    # Extract PATIENT-LEVEL dataframe
    maternal_patients = (
        maternal_data.get("patients", pd.DataFrame())
        if maternal_data
        else pd.DataFrame()
    )

    # Create cache key for summary data
    cache_key = (
        f"summary_facility_{facility_name}_{facility_uid}_{len(maternal_patients)}"
    )

    # Check if we have cached summary data
    if (
        cache_key in st.session_state.summary_kpi_cache_facility
        and time.time()
        - st.session_state.summary_kpi_cache_facility[cache_key]["timestamp"]
        < 300
    ):
        summary_data = st.session_state.summary_kpi_cache_facility[cache_key]["data"]
    else:
        # Calculate indicators from patient-level data
        with st.spinner("Computing summary statistics from patient-level data..."):
            # Get patient counts - THIS IS "Total Admitted"
            maternal_patient_count = count_unique_patients(
                maternal_patients, facility_uid
            )

            # Get earliest date
            maternal_start_date = get_earliest_date(
                maternal_patients, "enrollment_date"
            )

            # Compute maternal indicators using kpi_utils - ONLY NEED 2 RATES
            if facility_uid and "orgUnit" in maternal_patients.columns:
                filtered_maternal = maternal_patients[
                    maternal_patients["orgUnit"] == facility_uid
                ].copy()
            else:
                filtered_maternal = maternal_patients.copy()

            # Compute maternal KPIs
            maternal_kpis = compute_kpis(
                filtered_maternal, [facility_uid] if facility_uid else []
            )

            summary_data = {
                "maternal_tei_count": maternal_patient_count,  # Total Admitted
                "maternal_death_rate": maternal_kpis.get(
                    "maternal_death_rate", 0.0
                ),  # Maternal Death Rate
                "stillbirth_rate": maternal_kpis.get(
                    "stillbirth_rate", 0.0
                ),  # Stillbirth Rate
                "maternal_start_date": maternal_start_date,
                "location_name": location_name,
                "location_type": location_type,
            }

            # Cache the computed data
            st.session_state.summary_kpi_cache_facility[cache_key] = {
                "data": summary_data,
                "timestamp": time.time(),
            }

    # Extract data for rendering
    maternal_tei_count = summary_data["maternal_tei_count"]
    maternal_death_rate = summary_data["maternal_death_rate"]
    stillbirth_rate = summary_data["stillbirth_rate"]
    maternal_start_date = summary_data["maternal_start_date"]
    location_name = summary_data["location_name"]
    location_type = summary_data["location_type"]

    # SHOW ONLY THE 3 REQUIRED INDICATORS
    st.markdown("### 📊 Key Maternal Health Indicators")

    # Add location info above the metrics
    st.markdown(f"**📍 {location_type}: {location_name}**")
    if maternal_start_date != "N/A":
        st.markdown(f"**📅 Start Date: {maternal_start_date}**")

    col1, col2, col3 = st.columns(3)
    metrics = [
        (
            col1,
            "Total Admitted Mothers",
            f"{maternal_tei_count:,}",
            "Unique mothers admitted",
            "#1f77b4",
        ),
        (
            col2,
            "Maternal Death Rate",
            f"{maternal_death_rate:.2f}%",
            "Maternal mortality rate",
            "#d62728",
        ),
        (
            col3,
            "Stillbirth Rate",
            f"{stillbirth_rate:.2f}%",
            "Stillbirth rate",
            "#ff7f0e",
        ),
    ]

    for col, label, value, help_text, color in metrics:
        with col:
            st.markdown(
                f"""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid {color}; margin-bottom: 10px;">
                <div style="font-size: 0.8rem; color: #666; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {color}; margin: 10px 0;">{value}</div>
                <div style="font-size: 0.65rem; color: #888;">{help_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Simple summary table with only the 3 indicators
    st.markdown("### 📋 Summary Overview")

    # Create summary table data - Use consistent data types (all strings)
    summary_table_data = {
        "No": ["1", "2", "3", "4", "5"],  # All strings to avoid serialization issues
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Mothers",
            "Maternal Death Rate (%)",
            "Stillbirth Rate (%)",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_death_rate:.2f}%",
            f"{stillbirth_rate:.2f}%",
        ],
    }

    summary_table_df = pd.DataFrame(summary_table_data)

    # Use st.dataframe with all columns as strings to avoid Arrow serialization issues
    display_df = summary_table_df.astype(str)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button for summary data
    st.download_button(
        "📥 Download Summary Data",
        data=summary_table_df.to_csv(index=False),
        file_name=f"summary_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_maternal_dashboard_facility(
    user,
    maternal_data,
    facility_name,
    facility_uid,
    view_mode="Normal Trend",
):
    """Optimized Maternal Dashboard rendering using patient-level data with UID filtering"""

    # Only run if this is the active tab
    if st.session_state.active_tab != "maternal":
        return

    logging.info("Facility maternal dashboard rendering with patient-level data")

    if not maternal_data:
        st.error("No maternal data available")
        return

    # GET PATIENT-LEVEL DATA
    patients_df = maternal_data.get("patients", pd.DataFrame())

    if patients_df.empty:
        st.error("No patient data available")
        return

    # Ensure orgUnit column exists
    if "orgUnit" not in patients_df.columns:
        st.error("❌ Missing 'orgUnit' column in data. Cannot filter by facility UID.")
        return

    # Use patient data directly
    working_df = patients_df.copy()

    # Filter by UID
    if facility_uid and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"] == facility_uid].copy()

    # IMPORTANT: Use normalize_patient_dates to get proper event_date
    working_df = normalize_patient_dates(working_df)

    # Store the original df for KPI calculations
    st.session_state.maternal_patients_df = working_df.copy()

    # Update session state
    st.session_state.current_facility_uids = [facility_uid] if facility_uid else []
    st.session_state.current_display_names = [facility_name]
    st.session_state.current_comparison_mode = "facility"

    # Optimized header rendering
    header_title = f"🤰 Maternal Inpatient Data - {facility_name}"
    header_subtitle = "Single Facility View"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Displaying data from {header_subtitle}**")

    # Create containers for better performance
    kpi_container = st.container()

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_patient_filter_controls(
            working_df, container=col_ctrl, context="facility_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply date filters FIRST to get the correct time period
    filtered_for_all = apply_patient_filters(
        working_df, filters, [facility_uid] if facility_uid else []
    )

    # Store BOTH versions
    st.session_state["filtered_patients"] = filtered_for_all.copy()
    st.session_state["all_patients_for_kpi"] = filtered_for_all.copy()

    # Charts section
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        selected_kpi = render_kpi_tab_navigation()

        if view_mode == "Facility Comparison":
            # For facility dashboard, we don't have multiple facilities to compare
            st.info(
                "⚠️ Facility comparison requires multiple facilities. Using trend view instead."
            )
            view_mode = "Normal Trend"

        render_trend_chart_section(
            selected_kpi,
            filtered_for_all,
            [facility_uid] if facility_uid else [],
            [facility_name],
            bg_color,
            text_color,
            comparison_mode="facility",
            facilities_by_region=None,
            region_names=None,
        )

        render_additional_analytics(
            selected_kpi,
            filtered_for_all,
            [facility_uid] if facility_uid else [],
            bg_color,
            text_color,
        )


def render():
    """Main optimized render function for facility dashboard"""
    st.set_page_config(
        page_title="Facility Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Re-initialize session state for safety
    initialize_session_state_facility()

    # Show user change notification if needed
    if st.session_state.get("user_changed_facility", False):
        st.sidebar.info("User changed - loading fresh data...")
        current_user = st.session_state.get("user", {})
        clear_shared_cache_facility(current_user)

    # Load optimized CSS
    st.markdown(
        """
    <style>
    .main-header { font-size: 1.5rem !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; }
    .section-header { font-size: 1.2rem !important; margin: 0.2rem 0 !important; padding: 0.3rem 0 !important; }
    .stMarkdown { margin-bottom: 0.1rem !important; }
    .element-container { margin-bottom: 0.2rem !important; }
    .stButton button { margin-bottom: 0.2rem !important; }
    .stRadio > div { padding-top: 0.1rem !important; padding-bottom: 0.1rem !important; }
    .stForm { margin-bottom: 0.3rem !important; }
    hr { margin: 0.3rem 0 !important; }
    .user-info { margin-bottom: 0.3rem !important; font-size: 0.9rem; }
    .metric-card { min-height: 100px !important; padding: 12px !important; margin: 5px !important; }
    .metric-value { font-size: 1.5rem !important; margin: 5px 0 !important; }
    .metric-label { font-size: 0.8rem !important; margin-bottom: 2px !important; }
    .metric-help { font-size: 0.65rem !important; margin-top: 2px !important; }
    .summary-table th, .summary-table td { padding: 6px 8px !important; font-size: 12px !important; }
    .stCheckbox label { color: #000000 !important; font-size: 0.9rem !important; }
    .filter-box { background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Load additional CSS files if they exist
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Get user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    facility_name = user.get("facility_name", "Unknown Facility")
    facility_uid = user.get("facility_uid")

    # Compact sidebar user info
    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 {username}</div>
            <div>🏥 {facility_name}</div>
            <div>🛡️ {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh Data Button
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        clear_shared_cache_facility(user)
        st.session_state.refresh_trigger_facility = True
        st.session_state.selection_applied = True
        st.session_state.facility_filter_applied = False
        st.session_state.last_computed_kpis = None
        st.session_state.last_computed_newborn_kpis = None
        st.session_state.summary_kpi_cache_facility = {}
        st.session_state.data_initialized_facility = False

        # Clear data signature to force fresh computation
        current_user_key = (
            f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        )
        data_changed_key = f"data_signature_facility_{current_user_key}"
        if data_changed_key in st.session_state:
            del st.session_state[data_changed_key]

        # Reset tab states
        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
        st.session_state.tab_data_loaded["maternal"] = True
        st.session_state.tab_data_loaded["newborn"] = True
        st.session_state.tab_data_loaded["summary"] = False
        st.session_state.tab_loading["summary"] = False

        st.rerun()

    # ================ OPTIMIZED DATA LOADING ================
    # Get static data (cached for 1 hour)
    if not st.session_state.get(
        "static_data_loaded_facility", False
    ) or st.session_state.get("user_changed_facility", False):
        with st.sidebar:
            with st.spinner("Loading facility data..."):
                static_data = get_static_data_facility(user)
                st.session_state.facilities_facility = static_data["facilities"]
                st.session_state.facility_mapping_facility = static_data[
                    "facility_mapping"
                ]
                st.session_state.program_uid_map_facility = static_data[
                    "program_uid_map"
                ]
                st.session_state.static_data_loaded_facility = True
                st.session_state.user_changed_facility = False

    facilities = st.session_state.facilities_facility
    facility_mapping = st.session_state.facility_mapping_facility
    program_uid_map = st.session_state.program_uid_map_facility

    # SINGLE DATA LOADING - Only load once and store in variable
    if not st.session_state.get("data_initialized_facility", False):
        # First time or fresh data needed
        with st.spinner("Loading dashboard data..."):
            shared_data = get_shared_program_data_facility(
                user, program_uid_map, show_spinner=False
            )
            st.session_state.data_initialized_facility = True
            st.session_state.cached_shared_data_facility = shared_data
            logging.info("Facility initial data loading complete")
    else:
        # Use cached data from session state - no loading needed
        shared_data = st.session_state.cached_shared_data_facility
        logging.info("Using cached shared data from session state for facility")

    # Add cache status indicator in sidebar
    if st.session_state.get("data_initialized_facility", False):
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        timestamp_key = f"shared_data_timestamp_facility_{user_key}"

        if timestamp_key in st.session_state:
            time_elapsed = time.time() - st.session_state[timestamp_key]
            minutes_old = int(time_elapsed // 60)
            seconds_old = int(time_elapsed % 60)

            if minutes_old < 30:
                st.sidebar.info(f"Data: {minutes_old}m {seconds_old}s old")
            else:
                st.sidebar.warning(f"Data: {minutes_old}m old (will auto-refresh)")

    # ================ COMPACT VIEW MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">📊 View Mode</p>',
        unsafe_allow_html=True,
    )

    view_mode = "Normal Trend"

    # ================ OPTIMIZED TABS ================
    tab1, tab2, tab3 = st.tabs(
        [
            "🤰 **Maternal**",
            "👶 **Newborn**",
            "📊 **Summary**",
        ]
    )

    with tab1:
        # Update active tab only if this tab is clicked
        if st.session_state.active_tab != "maternal":
            st.session_state.active_tab = "maternal"
            logging.info("Switched to Maternal tab")

        maternal_data = shared_data["maternal"]
        if maternal_data:
            render_maternal_dashboard_facility(
                user,
                maternal_data,
                facility_name,
                facility_uid,
                view_mode=view_mode,
            )
        else:
            st.error("Maternal data not available")

    with tab2:
        if st.session_state.active_tab != "newborn":
            st.session_state.active_tab = "newborn"
            logging.info("Switched to Newborn tab")

        # Show maintenance message instead of actual newborn dashboard
        render_newborn_maintenance_message()

    with tab3:
        if st.session_state.active_tab != "summary":
            st.session_state.active_tab = "summary"
            logging.info("Switched to Summary tab")

        render_summary_dashboard_facility(
            user,
            facility_name,
            facility_uid,
            shared_data,
        )

    # Log current active tab state
    logging.info(f"Current active tab: {st.session_state.active_tab}")

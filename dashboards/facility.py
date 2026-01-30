# dashboards/facility.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from utils.resource import render_resources_tab
from newborns_dashboard.facility_newborn import (
    render_newborn_dashboard_shared,
)
from utils.kpi_utils import clear_cache
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_for_user,
    get_facility_mapping_for_user,
)
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
from utils.odk_dashboard import display_odk_dashboard

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
        "selected_facilities": [],  # Empty for facility level (single facility)
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
            "mentorship": False,
        },
        "tab_data_loaded": {
            "maternal": True,
            "newborn": True,
            "summary": False,
            "mentorship": False,
        },
        "tab_loading": {
            "summary": False,
            "mentorship": False,
        },
        "show_summarized_data": True,
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
        st.session_state.tab_data_loaded["mentorship"] = False
        st.session_state.tab_loading["summary"] = False
        st.session_state.tab_loading["mentorship"] = False
        st.session_state.show_summarized_data = True
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


def render_summary_dashboard_shared(
    user, facility_name, facility_mapping, selected_facilities, shared_data
):
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
                Get overview of key maternal and newborn health indicators for this facility
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

    # For facility level, we only have one facility
    facility_uids = (
        [facility_mapping.get(facility_name)]
        if facility_name in facility_mapping
        else []
    )
    display_names = [facility_name]

    # Use both maternal and newborn data
    maternal_data = shared_data["maternal"]
    newborn_data = shared_data["newborn"]

    if not maternal_data and not newborn_data:
        st.error("No data available for summary dashboard")
        return

    # Get patient dataframes
    maternal_patients = (
        maternal_data.get("patients", pd.DataFrame())
        if maternal_data
        else pd.DataFrame()
    )
    newborn_patients = (
        newborn_data.get("patients", pd.DataFrame()) if newborn_data else pd.DataFrame()
    )

    # Create cache key for summary data
    cache_key = f"summary_{location_name}_{len(facility_uids)}_{len(maternal_patients)}_{len(newborn_patients)}"

    # Check if we have cached summary data
    if (
        cache_key in st.session_state.summary_kpi_cache_facility
        and time.time()
        - st.session_state.summary_kpi_cache_facility[cache_key]["timestamp"]
        < 300
    ):
        summary_data = st.session_state.summary_kpi_cache_facility[cache_key]["data"]
    else:
        # Compute summary data using patient-level data
        with st.spinner("Computing summary statistics from patient-level data..."):
            # Get TEI counts from patient data
            maternal_patient_count = count_unique_patients(
                maternal_patients, facility_uids[0] if facility_uids else None
            )
            newborn_patient_count = count_unique_patients(
                newborn_patients, facility_uids[0] if facility_uids else None
            )

            # Get enrollment dates
            maternal_start_date = "N/A"
            newborn_start_date = "N/A"

            if "enrollment_date" in maternal_patients.columns:
                maternal_start_date = get_earliest_date(
                    maternal_patients, "enrollment_date"
                )

            if "enrollment_date" in newborn_patients.columns:
                newborn_start_date = get_earliest_date(
                    newborn_patients, "enrollment_date"
                )

            # Compute maternal indicators
            maternal_kpis = {}
            if not maternal_patients.empty and "orgUnit" in maternal_patients.columns:
                filtered_maternal = maternal_patients[
                    maternal_patients["orgUnit"].isin(facility_uids)
                ].copy()
                from utils.kpi_utils import compute_kpis

                maternal_kpis = compute_kpis(filtered_maternal, facility_uids)

            # Compute newborn indicators - ONLY THE 3 REQUIRED ONES
            newborn_kpis = {}
            if not newborn_patients.empty and "orgUnit" in newborn_patients.columns:
                filtered_newborn = newborn_patients[
                    newborn_patients["orgUnit"].isin(facility_uids)
                ].copy()

                # Import only the 3 required newborn KPI functions
                from newborns_dashboard.kpi_utils_newborn import (
                    compute_admitted_newborns_count,  # For Total Admitted Newborns
                    compute_neonatal_mortality_rate,  # For NMR
                    compute_hypothermia_on_admission_rate,  # For Hypothermia on Admission
                )

                # Compute only the 3 required newborn indicators:
                # 1. Total Admitted Newborns (using enrollment date)
                admitted_newborns_count = compute_admitted_newborns_count(
                    filtered_newborn, facility_uids
                )

                # 2. Neonatal Mortality Rate (NMR)
                nmr_rate, death_count, total_deaths = compute_neonatal_mortality_rate(
                    filtered_newborn, facility_uids
                )

                # 3. Hypothermia After Admission Rate
                hypothermia_rate, hypo_count, total_hypo = (
                    compute_hypothermia_on_admission_rate(
                        filtered_newborn, facility_uids
                    )
                )

                newborn_kpis = {
                    "admitted_newborns_count": admitted_newborns_count,
                    "neonatal_mortality_rate": nmr_rate,
                    "hypothermia_rate": hypothermia_rate,
                    "death_count": death_count,
                    "total_deaths": total_deaths,
                    "hypothermia_count": hypo_count,
                    "total_hypo": total_hypo,
                }

            # Store only the required indicators for each program
            summary_data = {
                # Maternal indicators (keep as is)
                "maternal_tei_count": maternal_patient_count,  # Total Admitted Mothers
                "maternal_death_rate": maternal_kpis.get("maternal_death_rate", 0.0),
                "stillbirth_rate": maternal_kpis.get("stillbirth_rate", 0.0),
                "maternal_start_date": maternal_start_date,
                # Newborn indicators - ONLY 3 REQUIRED:
                "newborn_tei_count": newborn_kpis.get(
                    "admitted_newborns_count", 0
                ),  # 1. Total Admitted Newborns
                "neonatal_mortality_rate": newborn_kpis.get(
                    "neonatal_mortality_rate", 0.0
                ),  # 2. NMR
                "hypothermia_rate": newborn_kpis.get(
                    "hypothermia_rate", 0.0
                ),  # 3. Hypothermia After Admission
                "newborn_start_date": newborn_start_date,
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

    newborn_tei_count = summary_data["newborn_tei_count"]
    neonatal_mortality_rate = summary_data["neonatal_mortality_rate"]
    hypothermia_rate = summary_data["hypothermia_rate"]
    newborn_start_date = summary_data["newborn_start_date"]

    location_name = summary_data["location_name"]
    location_type = summary_data["location_type"]

    # SHOW ONLY THE 3 REQUIRED INDICATORS FOR EACH PROGRAM
    st.markdown("### 📊 Key Maternal Health Indicators")

    # Add location info above the metrics
    st.markdown(f"**📍 {location_type}: {location_name}**")
    if maternal_start_date != "N/A":
        st.markdown(f"**📅 Start Date: {maternal_start_date}**")

    col1, col2, col3 = st.columns(3)
    maternal_metrics = [
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

    for col, label, value, help_text, color in maternal_metrics:
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

    st.markdown("### 👶 Key Newborn Health Indicators")

    if newborn_start_date != "N/A":
        st.markdown(f"**📅 Start Date: {newborn_start_date}**")

    col1, col2, col3 = st.columns(3)
    newborn_metrics = [
        (
            col1,
            "Total Admitted Newborns",
            f"{newborn_tei_count:,}",
            "Unique newborns admitted (enrollment date)",
            "#1f77b4",
        ),
        (
            col2,
            "Neonatal Mortality Rate",
            f"{neonatal_mortality_rate:.2f}%",
            "Newborn mortality rate (NMR)",
            "#d62728",
        ),
        (
            col3,
            "Hypothermia on Admission",
            f"{hypothermia_rate:.2f}%",
            "Newborns with temp < 36.5°C on admission",
            "#2ca02c",
        ),
    ]

    for col, label, value, help_text, color in newborn_metrics:
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


def render_maternal_dashboard_shared(
    user,
    maternal_data,
    facility_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",
):
    """Optimized Maternal Dashboard rendering using patient-level data with UID filtering - FACILITY VERSION"""

    # Only run if this is the active tab
    if st.session_state.active_tab != "maternal":
        return

    logging.info("Maternal dashboard rendering with patient-level data - FACILITY")

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
        st.error("❌ Missing 'orgUnit' column in data. Cannot filter by facility UIDs.")
        return

    # For facility level, we only have one facility
    facility_uid = (
        facility_mapping.get(facility_name)
        if facility_name in facility_mapping
        else None
    )
    facility_uids = [facility_uid] if facility_uid else []
    display_names = [facility_name]
    comparison_mode = "facility"

    # Use patient data directly
    working_df = patients_df.copy()

    # Filter by UID EARLY
    if facility_uid and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"] == facility_uid].copy()
        logging.info(
            f"✅ FACILITY: Filtered by facility UID: {len(working_df)} patients remain"
        )
    else:
        logging.info(
            f"⚠️ FACILITY: No facility UID or orgUnit column. Keeping all {len(working_df)} patients"
        )

    # =========== CRITICAL: USE normalize_patient_dates ===========
    working_df = normalize_patient_dates(working_df)

    if working_df.empty:
        st.info("ℹ️ No data available for the selected facility/period.")
        return

    # Log date statistics
    valid_dates = working_df["enrollment_date"].notna().sum()
    total_patients = len(working_df)
    logging.info(
        f"📅 FACILITY: enrollment_date - {valid_dates}/{total_patients} valid dates"
    )

    # Log sample dates to verify they're correct
    if valid_dates > 0:
        sample_dates = working_df["enrollment_date"].dropna().head(3).tolist()
        logging.info(f"📅 FACILITY: Sample dates: {sample_dates}")
    # =========== END OF CRITICAL ADDITION ===========

    # Get current KPI selection
    current_kpi = st.session_state.get(
        "selected_kpi", "Maternal Death Rate (per 100,000)"
    )
    from utils.kpi_utils import get_relevant_date_column_for_kpi

    kpi_date_column = get_relevant_date_column_for_kpi(current_kpi)

    # Store the original df for KPI calculations
    st.session_state.maternal_patients_df = working_df.copy()

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # Optimized header rendering
    header_title = f"🤰 Maternal Inpatient Data - {facility_name}"
    header_subtitle = "Single Facility View"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.1rem; line-height: 1.2;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"<div style='margin-bottom: 0.1rem;'><b>Displaying data from {header_subtitle}</b></div>", unsafe_allow_html=True)

    # Progress container
    # progress_container removed for layout stability
    # with progress_container.container():
    #     st.markdown("---")
    #     st.markdown("### 📈 Preparing Dashboard...")
    # progress_indicator logic removed

    # kpi_container = st.container() # Removed empty container

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_patient_filter_controls(
            working_df, container=col_ctrl, context="facility_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply date filters FIRST to get the correct time period
    logging.info(
        f"🔍 FACILITY: Calling apply_patient_filters with {len(working_df)} patients"
    )
    filtered_for_all = apply_patient_filters(working_df, filters, facility_uids)

    logging.info(
        f"🔍 FACILITY: After apply_patient_filters: {len(filtered_for_all)} patients"
    )

    # Store BOTH versions
    st.session_state["filtered_patients"] = filtered_for_all.copy()
    st.session_state["all_patients_for_kpi"] = filtered_for_all.copy()

    # progress_container.empty() removed

    # Charts section
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        selected_kpi = render_kpi_tab_navigation()

        # For facility level, always use trend view (no comparison possible)
        render_trend_chart_section(
            selected_kpi,
            filtered_for_all,
            facility_uids,
            display_names,
            bg_color,
            text_color,
            comparison_mode=comparison_mode,
            facilities_by_region=facilities_by_region,
            region_names=display_names if comparison_mode == "region" else None,
        )

        render_additional_analytics(
            selected_kpi,
            filtered_for_all,
            facility_uids,
            bg_color,
            text_color,
        )


def render():
    """Main optimized render function for facility dashboard"""
    # st.set_page_config removed - only app.py should call this

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

        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
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
            # Sidebar info removed per user request

    # ================ OPTIMIZED TABS WITH PROPER ISOLATION ================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🤰 **Maternal**",
            "👶 **Newborn**",
            "📊 **Summary**",
            "📋 **Mentorship**",
            "📚 **Resources**",
        ]
    )

    with tab1:
        # Update active tab only if this tab is clicked
        if st.session_state.active_tab != "maternal":
            st.session_state.active_tab = "maternal"
            logging.info("Switched to Maternal tab")

        maternal_data = shared_data["maternal"]
        if maternal_data:
            # Create dummy facilities_by_region for compatibility
            facilities_by_region = {
                facility_name: [(facility_name, facility_mapping.get(facility_name))]
            }

            render_maternal_dashboard_shared(
                user,
                maternal_data,
                facility_name,
                facilities_by_region,
                facility_mapping,
                view_mode="Normal Trend",  # Facility has only one facility, so no comparison
            )
        else:
            st.error("Maternal data not available")

    with tab2:
        if st.session_state.active_tab != "newborn":
            st.session_state.active_tab = "newborn"
            logging.info("Switched to Newborn tab")

        newborn_data = shared_data["newborn"]
        if newborn_data:
            # Create dummy facilities_by_region for compatibility
            facilities_by_region = {
                facility_name: [(facility_name, facility_mapping.get(facility_name))]
            }

            render_newborn_dashboard_shared(
                user,
                newborn_data,
                facility_name,
                facilities_by_region,
                facility_mapping,
                view_mode="Normal Trend",  # Facility has only one facility, so no comparison
            )
        else:
            st.error("Newborn data not available")

    with tab3:
        if st.session_state.active_tab != "summary":
            st.session_state.active_tab = "summary"
            logging.info("Switched to Summary tab")

        # Empty selected_facilities list for facility level
        selected_facilities = []

        render_summary_dashboard_shared(
            user,
            facility_name,
            facility_mapping,
            selected_facilities,
            shared_data,
        )

    with tab4:
        if st.session_state.active_tab != "mentorship":
            st.session_state.active_tab = "mentorship"
            logging.info("Switched to Mentorship tab")

        # Check if mentorship data should be loaded
        if not st.session_state.tab_data_loaded["mentorship"]:
            st.markdown(
                """
            <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                 border-radius: 12px; border: 2px dashed #dee2e6; margin: 2rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
                <h2 style="color: #495057; margin-bottom: 1rem;">Mentorship Dashboard</h2>
                <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                    View mentorship tracking data and ODK form submissions
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "Load Mentorship Data",
                    use_container_width=True,
                    type="primary",
                    key="load_mentorship_data_facility",
                ):
                    st.session_state.tab_loading["mentorship"] = True
                    st.session_state.tab_data_loaded["mentorship"] = True
                    st.rerun()
        else:
            # Show loading indicator if data is being processed
            if st.session_state.tab_loading["mentorship"]:
                st.markdown(
                    """
                <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                     border-radius: 12px; border: 2px solid #dee2e6; margin: 2rem 0;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
                    <h2 style="color: #495057; margin-bottom: 1rem;">Loading Mentorship Dashboard...</h2>
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
                st.session_state.tab_loading["mentorship"] = False
                st.rerun()
            display_odk_dashboard(user)
    with tab5:
        if st.session_state.active_tab != "resources":
            st.session_state.active_tab = "resources"
            logging.info("Switched to Resources tab")
    
        render_resources_tab()
    # Log current active tab state
    logging.info(f"Current active tab: {st.session_state.active_tab}")

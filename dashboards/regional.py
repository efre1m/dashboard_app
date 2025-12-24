# dashboards/regional.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from components.kpi_card import render_kpi_cards
from newborns_dashboard.reginal_newborn import render_newborn_dashboard
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_for_user,
    get_facility_mapping_for_user,
)
from utils.dash_co import (
    normalize_patient_dates,  # Changed from normalize_event_dates
    normalize_enrollment_dates,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
    apply_patient_filters,  # Changed from apply_simple_filters
    render_patient_filter_controls,  # Changed from render_simple_filter_controls
    render_kpi_tab_navigation,
)
from utils.kpi_utils import clear_cache, compute_kpis
from utils.kpi_lbw import compute_lbw_kpi
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)
from utils.odk_dashboard import display_odk_dashboard
from dashboards.data_quality_tracking import render_data_quality_tracking
from newborns_dashboard.kpi_nmr import compute_nmr_kpi

# Initialize status system
initialize_status_system()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# Performance optimization: Pre-load essential data with user-specific caching
@st.cache_data(ttl=3600, show_spinner=False)
def get_static_data(user):
    """Cache static data that doesn't change often - user-specific"""
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


# Optimized shared cache with user-specific keys
@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=5)
def fetch_shared_program_data(user, program_uid):
    """Optimized shared cache for program data - user-specific"""
    if not program_uid:
        return None
    try:
        return fetch_program_data_for_user(user, program_uid)
    except Exception as e:
        logging.error(f"Error fetching data for program {program_uid}: {e}")
        return None


def get_shared_program_data_optimized(user, program_uid_map, show_spinner=True):
    """Smart data loading with 30-minute auto-refresh and user-specific caching"""
    maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
    newborn_program_uid = program_uid_map.get("Newborn Care Form")

    # Create user-specific session state keys
    user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    shared_loaded_key = f"shared_data_loaded_{user_key}"
    shared_maternal_key = f"shared_maternal_data_{user_key}"
    shared_newborn_key = f"shared_newborn_data_{user_key}"
    shared_timestamp_key = f"shared_data_timestamp_{user_key}"

    current_time = time.time()

    # ✅ Check if cache has expired (30 minutes = 1800 seconds)
    cache_expired = False
    if shared_timestamp_key in st.session_state:
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        cache_expired = time_elapsed > 1800  # 30 minutes
        if cache_expired:
            logging.info(
                f"🔄 Cache expired after {time_elapsed:.0f} seconds, fetching fresh data"
            )

    # ✅ Check if user changed (force fresh data)
    user_changed = st.session_state.get("user_changed", False)

    # ✅ Determine if we need fresh data
    need_fresh_data = (
        not st.session_state.get(shared_loaded_key, False)  # First load
        or cache_expired  # Cache expired
        or user_changed  # User changed
        or st.session_state.get("refresh_trigger", False)  # Manual refresh
    )

    if need_fresh_data:
        logging.info(
            "🔄 Fetching FRESH data (cache expired, user changed, or manual refresh)"
        )

        # Clear existing cache
        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None

        # Load fresh data in parallel - ONLY show spinner if requested
        spinner_text = "🚀 Loading dashboard data..." if show_spinner else None

        def load_data():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                maternal_future = (
                    executor.submit(
                        fetch_shared_program_data, user, maternal_program_uid
                    )
                    if maternal_program_uid
                    else None
                )
                newborn_future = (
                    executor.submit(
                        fetch_shared_program_data, user, newborn_program_uid
                    )
                    if newborn_program_uid
                    else None
                )

                # Get results with timeout
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
                    st.session_state[shared_timestamp_key] = (
                        current_time  # Update timestamp
                    )

                    # ✅ STORE in main cached_shared_data for easy access
                    st.session_state.cached_shared_data = {
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
                        f"✅ FRESH DATA: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients"
                    )

                    # Reset refresh trigger and user changed flags
                    st.session_state.refresh_trigger = False
                    st.session_state.user_changed = False

                except concurrent.futures.TimeoutError:
                    logging.error("Data loading timeout")
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
        # ✅ Use cached data
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
            f"✅ USING CACHED DATA: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients ({time_elapsed:.0f}s old)"
        )

    return {
        "maternal": st.session_state[shared_maternal_key],
        "newborn": st.session_state[shared_newborn_key],
    }


def clear_shared_cache(user=None):
    """Clear shared data cache - user-specific"""
    if user:
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        shared_loaded_key = f"shared_data_loaded_{user_key}"
        shared_maternal_key = f"shared_maternal_data_{user_key}"
        shared_newborn_key = f"shared_newborn_data_{user_key}"
        shared_timestamp_key = f"shared_data_timestamp_{user_key}"

        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None
        if shared_timestamp_key in st.session_state:
            del st.session_state[shared_timestamp_key]

        logging.info("🧹 Cleared user-specific cache")
    else:
        # Clear all user caches
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith("shared_data_loaded_")
            or key.startswith("shared_maternal_data_")
            or key.startswith("shared_newborn_data_")
            or key.startswith("shared_data_timestamp_")
        ]
        for key in keys_to_clear:
            del st.session_state[key]
        logging.info("🧹 Cleared ALL shared caches")

    clear_cache()


def initialize_session_state():
    """Optimized session state initialization with proper tab isolation"""
    session_vars = {
        "refresh_trigger": False,  # ✅ Added for manual refresh tracking
        "selected_facilities": ["All Facilities"],
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
        "filtered_patients": pd.DataFrame(),  # Changed from filtered_events
        "selection_applied": True,
        "cached_patients_data": None,  # Changed from cached_events_data
        "cached_enrollments_data": None,
        "cached_tei_data": None,
        "last_applied_selection": "All Facilities",
        "kpi_cache": {},
        "selected_program_uid": None,
        "selected_program_name": "Maternal Inpatient Data",
        "static_data_loaded": False,
        "facility_filter_applied": False,
        "current_user_identifier": None,
        "user_changed": False,
        # KPI sharing
        "last_computed_kpis": None,
        "last_computed_facilities": None,
        "last_computed_timestamp": None,
        "last_computed_newborn_kpis": None,
        "last_computed_newborn_timestamp": None,
        "summary_kpi_cache": {},
        # ✅ FIXED: Proper tab tracking
        "active_tab": "maternal",
        "data_initialized": False,
        "tab_initialized": {
            "maternal": False,
            "newborn": False,
            "summary": False,
            "mentorship": False,
            "data_quality": False,
        },
        # ✅ NEW: Track which tabs have been activated by user
        "tab_data_loaded": {
            "maternal": True,  # Always load maternal and newborn by default
            "newborn": True,  # Always load newborn by default
            "summary": False,  # Only load when user clicks button
            "mentorship": False,  # Only load when user clicks button
            "data_quality": True,  # Always load data quality by default
        },
        # ✅ NEW: Track loading state for each tab
        "tab_loading": {
            "summary": False,
            "mentorship": False,
        },
        # ✅ NEW: Force show tables in summary dashboard
        "show_summarized_data_regional": True,
        # Regional specific
        "facilities": [],
        "facility_mapping": {},
        "program_uid_map": {},
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Check if user has changed
    current_user = st.session_state.get("user", {})
    new_user_identifier = f"{current_user.get('username', 'unknown')}_{current_user.get('role', 'unknown')}"

    if st.session_state.current_user_identifier != new_user_identifier:
        st.session_state.user_changed = True
        st.session_state.current_user_identifier = new_user_identifier
        # Clear user-specific data when user changes
        st.session_state.static_data_loaded = False
        st.session_state.selected_facilities = ["All Facilities"]
        st.session_state.selection_applied = True
        st.session_state.facility_filter_applied = False
        st.session_state.last_computed_kpis = None
        st.session_state.last_computed_newborn_kpis = None
        st.session_state.summary_kpi_cache = {}
        st.session_state.data_initialized = False
        # Reset all tab states
        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
        # Reset data loaded flags except for default tabs
        st.session_state.tab_data_loaded["maternal"] = True
        st.session_state.tab_data_loaded["newborn"] = True
        st.session_state.tab_data_loaded["summary"] = False
        st.session_state.tab_data_loaded["mentorship"] = False
        st.session_state.tab_data_loaded["data_quality"] = True
        # Reset loading states
        st.session_state.tab_loading["summary"] = False
        st.session_state.tab_loading["mentorship"] = False
        # Reset show tables
        st.session_state.show_summarized_data_regional = True
    else:
        st.session_state.user_changed = False


# Initialize session state at the very beginning
initialize_session_state()


def count_unique_patients(patient_df, facility_uids, org_unit_column="orgUnit"):
    """✅ Count unique patients from patient-level dataframe using UIDs"""
    if patient_df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and org_unit_column in patient_df.columns:
        filtered_patients = patient_df[patient_df[org_unit_column].isin(facility_uids)]
    else:
        filtered_patients = patient_df

    # Count unique TEI IDs
    if "tei_id" in filtered_patients.columns:
        count = filtered_patients["tei_id"].nunique()
        print(f"\n🔍 COUNT_UNIQUE_PATIENTS:")
        print(f"   Total patients: {len(filtered_patients)}")
        print(f"   Unique TEI IDs: {count}")
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


def calculate_maternal_indicators_from_patients(patient_df, facility_uids):
    """✅ Calculate maternal indicators from patient-level data"""
    if patient_df.empty:
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

    # Filter by facilities if specified
    if facility_uids and "orgUnit" in patient_df.columns:
        filtered_df = patient_df[patient_df["orgUnit"].isin(facility_uids)]
    else:
        filtered_df = patient_df

    # Count total patients (each patient is a delivery)
    total_deliveries = filtered_df["tei_id"].nunique()

    # Initialize counts
    maternal_deaths = 0
    live_births = 0
    stillbirths = 0
    low_birth_weight_count = 0
    total_weighed = 0

    # Check for maternal death indicators in patient data
    if "delivery_summary_maternal_death" in filtered_df.columns:
        maternal_deaths = filtered_df["delivery_summary_maternal_death"].sum()

    # Count live births and stillbirths
    if "delivery_summary_live_birth" in filtered_df.columns:
        live_births = filtered_df["delivery_summary_live_birth"].sum()
    if "delivery_summary_stillbirth" in filtered_df.columns:
        stillbirths = filtered_df["delivery_summary_stillbirth"].sum()

    total_births = live_births + stillbirths

    # Calculate low birth weight
    if "delivery_summary_birth_weight" in filtered_df.columns:
        # Filter out null weights
        weight_df = filtered_df[filtered_df["delivery_summary_birth_weight"].notna()]
        total_weighed = len(weight_df)
        if total_weighed > 0:
            # Convert to numeric
            weight_df["delivery_summary_birth_weight"] = pd.to_numeric(
                weight_df["delivery_summary_birth_weight"], errors="coerce"
            )
            low_birth_weight_count = weight_df[
                weight_df["delivery_summary_birth_weight"] < 2500
            ].shape[0]
            low_birth_weight_rate = (
                (low_birth_weight_count / total_weighed * 100)
                if total_weighed > 0
                else 0
            )
        else:
            low_birth_weight_rate = 0
    else:
        low_birth_weight_rate = 0

    # Calculate rates
    maternal_death_rate = (
        (maternal_deaths / total_deliveries * 100000) if total_deliveries > 0 else 0
    )
    stillbirth_rate = (stillbirths / total_births * 1000) if total_births > 0 else 0

    return {
        "total_deliveries": total_deliveries,
        "maternal_deaths": int(maternal_deaths),
        "maternal_death_rate": round(maternal_death_rate, 2),
        "live_births": int(live_births),
        "stillbirths": int(stillbirths),
        "total_births": int(total_births),
        "stillbirth_rate": round(stillbirth_rate, 2),
        "low_birth_weight_rate": round(low_birth_weight_rate, 2),
        "low_birth_weight_count": int(low_birth_weight_count),
    }


def calculate_newborn_indicators_from_patients(patient_df, facility_uids):
    """✅ Calculate newborn indicators from patient-level data"""
    if patient_df.empty:
        return {
            "total_admitted": 0,
            "nmr": 0.0,
            "nmr_dead_count": 0,
            "nmr_total_admitted": 0,
            "kmc_coverage_rate": 0.0,
            "kmc_cases": 0,
            "total_lbw": 0,
        }

    # Filter by facilities if specified
    if facility_uids and "orgUnit" in patient_df.columns:
        filtered_df = patient_df[patient_df["orgUnit"].isin(facility_uids)]
    else:
        filtered_df = patient_df

    # Count total admitted newborns
    total_admitted = filtered_df["tei_id"].nunique()

    # Calculate NMR from discharge status
    nmr_dead_count = 0
    if (
        "discharge_and_final_diagnosis_newborn_status_at_discharge"
        in filtered_df.columns
    ):
        dead_cases = filtered_df[
            filtered_df["discharge_and_final_diagnosis_newborn_status_at_discharge"]
            == "dead"
        ]
        nmr_dead_count = len(dead_cases)

    nmr_rate = (nmr_dead_count / total_admitted * 1000) if total_admitted > 0 else 0

    # Calculate KMC coverage
    kmc_cases = 0
    if "interventions_kmc_administered" in filtered_df.columns:
        kmc_cases = filtered_df[
            filtered_df["interventions_kmc_administered"] == 1
        ].shape[0]
    kmc_coverage_rate = (kmc_cases / total_admitted * 100) if total_admitted > 0 else 0

    # Count low birth weight newborns
    total_lbw = 0
    if "admission_information_weight_on_admission" in filtered_df.columns:
        weight_df = filtered_df[
            filtered_df["admission_information_weight_on_admission"].notna()
        ]
        if not weight_df.empty:
            weight_df["admission_information_weight_on_admission"] = pd.to_numeric(
                weight_df["admission_information_weight_on_admission"], errors="coerce"
            )
            total_lbw = weight_df[
                weight_df["admission_information_weight_on_admission"] < 2500
            ].shape[0]

    return {
        "total_admitted": total_admitted,
        "nmr": round(nmr_rate, 2),
        "nmr_raw": nmr_rate,
        "nmr_dead_count": nmr_dead_count,
        "nmr_total_admitted": total_admitted,
        "kmc_coverage_rate": round(kmc_coverage_rate, 2),
        "kmc_cases": kmc_cases,
        "total_lbw": total_lbw,
    }


def get_location_display_name(selected_facilities, region_name):
    """Optimized location display name generation"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        return ", ".join(selected_facilities), "Facilities"


def filter_patient_data_by_facilities(patient_df, facility_uids):
    """Filter patient dataframe by facility UIDs"""
    if not patient_df.empty and facility_uids and "orgUnit" in patient_df.columns:
        print(f"\n🔍 FILTER_PATIENT_DATA_BY_FACILITIES:")
        print(f"   Before filter: {len(patient_df)} patients")
        print(f"   Filtering by {len(facility_uids)} UIDs")

        filtered_df = patient_df[patient_df["orgUnit"].isin(facility_uids)]

        print(f"   After filter: {len(filtered_df)} patients")
        print(f"   Removed: {len(patient_df) - len(filtered_df)} patients")

        # Show breakdown
        if "orgUnit_name" in filtered_df.columns:
            for uid in facility_uids:
                uid_data = filtered_df[filtered_df["orgUnit"] == uid]
                if not uid_data.empty:
                    facility_name = uid_data["orgUnit_name"].iloc[0]
                    print(f"   UID {uid} ({facility_name}): {len(uid_data)} patients")

        return filtered_df
    return patient_df


def render_loading_indicator(tab_name, icon):
    """Render a loading indicator for tabs that are processing data"""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
             border-radius: 12px; border: 2px solid #dee2e6; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h2 style="color: #495057; margin-bottom: 1rem;">Loading {tab_name}...</h2>
            <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                Please wait while we process the data. This may take 1-2 minutes.
            </p>
            <div style="display: inline-block; padding: 10px 20px; background: #007bff; color: white; 
                 border-radius: 25px; font-weight: bold;">
                ⏳ Processing Data...
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Simulate processing time
    time.sleep(2)


def render_tab_placeholder(tab_name, icon, tab_key, description):
    """Render a placeholder with a button to load data for the tab"""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
             border-radius: 12px; border: 2px dashed #dee2e6; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h2 style="color: #495057; margin-bottom: 1rem;">{tab_name}</h2>
            <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
                {description}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"🚀 Load {tab_name} Data",
            use_container_width=True,
            type="primary",
            key=f"load_{tab_key}_data",
        ):
            # Set loading state and data loaded state
            st.session_state.tab_loading[tab_key] = True
            st.session_state.tab_data_loaded[tab_key] = True
            st.rerun()


def render_summary_dashboard_shared(
    user, region_name, facility_mapping, selected_facilities, shared_data
):
    """✅ OPTIMIZED Summary Dashboard using patient-level data"""

    # ✅ FIXED: Only run if this is the active tab AND data is loaded
    if st.session_state.active_tab != "summary":
        return

    # ✅ Check if user has clicked "View Data" button for this tab
    if not st.session_state.tab_data_loaded["summary"]:
        render_tab_placeholder(
            "Summary Dashboard",
            "📊",
            "summary",
            "Get comprehensive overview of maternal and newborn health indicators across selected facilities",
        )
        return

    # ✅ Show loading indicator if data is being processed
    if st.session_state.tab_loading["summary"]:
        render_loading_indicator("Summary Dashboard", "📊")
        # After loading, set loading to False
        st.session_state.tab_loading["summary"] = False
        st.rerun()

    logging.info("🔄 Regional summary dashboard rendering - USING PATIENT-LEVEL DATA")

    # Get location display name
    location_name, location_type = get_location_display_name(
        selected_facilities, region_name
    )

    # Get facility UIDs
    if selected_facilities == ["All Facilities"]:
        facility_uids = list(facility_mapping.values())
    else:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]

    # Use patient-level data
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
        cache_key in st.session_state.summary_kpi_cache
        and time.time() - st.session_state.summary_kpi_cache[cache_key]["timestamp"]
        < 300
    ):
        summary_data = st.session_state.summary_kpi_cache[cache_key]["data"]
        logging.info("✅ USING CACHED summary data")
    else:
        # ✅ Compute summary data using patient-level data
        with st.spinner("🔄 Computing summary statistics from patient-level data..."):
            # Get TEI counts from patient data
            maternal_patient_count = count_unique_patients(
                maternal_patients, facility_uids
            )
            newborn_patient_count = count_unique_patients(
                newborn_patients, facility_uids
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

            # ✅ Calculate maternal indicators from patient data
            maternal_indicators = calculate_maternal_indicators_from_patients(
                maternal_patients, facility_uids
            )

            # ✅ Calculate newborn indicators from patient data
            newborn_indicators = calculate_newborn_indicators_from_patients(
                newborn_patients, facility_uids
            )

            # ✅ Facility comparison using patient data
            facility_comparison_data = {}
            for facility_name, facility_uid in facility_mapping.items():
                # Count maternal patients
                if (
                    not maternal_patients.empty
                    and "orgUnit" in maternal_patients.columns
                ):
                    facility_maternal_patients = maternal_patients[
                        maternal_patients["orgUnit"] == facility_uid
                    ]
                    maternal_count = facility_maternal_patients["tei_id"].nunique()
                else:
                    maternal_count = 0

                # Count newborn patients
                if not newborn_patients.empty and "orgUnit" in newborn_patients.columns:
                    facility_newborn_patients = newborn_patients[
                        newborn_patients["orgUnit"] == facility_uid
                    ]
                    newborn_count = facility_newborn_patients["tei_id"].nunique()
                else:
                    newborn_count = 0

                # Only include facilities with data
                if maternal_count > 0 or newborn_count > 0:
                    short_name = shorten_facility_name(facility_name)
                    facility_comparison_data[short_name] = {
                        "mothers": maternal_count,
                        "newborns": newborn_count,
                        "full_name": facility_name,
                    }

            summary_data = {
                "maternal_indicators": maternal_indicators,
                "newborn_indicators": newborn_indicators,
                "maternal_tei_count": maternal_patient_count,
                "newborn_tei_count": newborn_patient_count,
                "newborn_start_date": newborn_start_date,
                "maternal_start_date": maternal_start_date,
                "facility_comparison_data": facility_comparison_data,
            }

            # Cache the computed data
            st.session_state.summary_kpi_cache[cache_key] = {
                "data": summary_data,
                "timestamp": time.time(),
            }

    # Extract data for rendering
    maternal_indicators = summary_data["maternal_indicators"]
    newborn_indicators = summary_data["newborn_indicators"]
    maternal_tei_count = summary_data["maternal_tei_count"]
    newborn_tei_count = summary_data["newborn_tei_count"]
    newborn_start_date = summary_data["newborn_start_date"]
    maternal_start_date = summary_data["maternal_start_date"]
    facility_comparison_data = summary_data["facility_comparison_data"]

    # Apply optimized table styling
    st.markdown(
        """
    <style>
    .summary-table-container { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0; border: 1px solid #e0e0e0; }
    .summary-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .summary-table thead tr { background: linear-gradient(135deg, #1f77b4, #1668a1); }
    .summary-table th { color: white; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 13px; border: none; }
    .summary-table td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; font-size: 13px; background-color: white; }
    .summary-table tbody tr:last-child td { border-bottom: none; }
    .summary-table tbody tr:hover td { background-color: #f8f9fa; }
    .newborn-table thead tr { background: linear-gradient(135deg, #1f77b4, #1668a1) !important; }
    .maternal-table thead tr { background: linear-gradient(135deg, #2ca02c, #228b22) !important; }
    .summary-table td:first-child { font-weight: 600; color: #666; text-align: center; }
    .summary-table th:first-child { text-align: center; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ✅ DIRECTLY SHOW TABLES WITHOUT SUMMARY BUTTON
    # Show quick statistics at the top
    st.markdown("### 📈 Quick Statistics")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (
            col1,
            "👩 Total Mothers",
            maternal_tei_count,
            "info-metric",
            "Unique mothers admitted",
        ),
        (
            col2,
            "👶 Total Newborns",
            newborn_tei_count,
            "success-metric",
            "Unique newborns admitted",
        ),
        (
            col3,
            "⚠️ Maternal Deaths",
            maternal_indicators["maternal_deaths"],
            "critical-metric",
            "Maternal mortality cases",
        ),
        (
            col4,
            "📅 Coverage Start",
            maternal_start_date,
            "warning-metric",
            "Earliest data record",
        ),
    ]

    for col, label, value, css_class, help_text in metrics:
        with col:
            st.markdown(
                f"""
            <div class="metric-card {css_class}" style="min-height: 120px; padding: 15px;">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value if isinstance(value, int) else value}</div>
                <div class="metric-help">{help_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ✅ DIRECTLY SHOW ALL TABLES (no button needed)
    st.markdown("---")

    # Newborn Overview Table
    st.markdown("### 👶 Newborn Care Overview")

    # Create newborn table data with actual NMR value
    newborn_table_data = {
        "No": list(range(1, 10)),
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Newborns",
            "NMR (per 1000)",
            "Stillbirth Rate (per 1000)",
            "Live Births",
            "Stillbirths",
            "Total Births",
            "Low Birth Weight Rate (<2500g)",
        ],
        "Value": [
            newborn_start_date,
            location_name,
            f"{newborn_tei_count:,}",
            f"{newborn_indicators['nmr']:.2f}",
            f"{maternal_indicators['stillbirth_rate']:.2f}",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['total_births']:,}",
            f"{maternal_indicators['low_birth_weight_rate']:.2f}%",
        ],
    }

    newborn_table_df = pd.DataFrame(newborn_table_data)
    st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
    st.markdown(
        newborn_table_df.style.set_table_attributes(
            'class="summary-table newborn-table"'
        )
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .to_html(),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Download button for newborn data with NMR
    col_info, col_download = st.columns([3, 1])
    with col_download:
        newborn_data_download = {
            "Start Date": [newborn_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Newborns": [newborn_tei_count],
            "NMR (per 1000)": [f"{newborn_indicators['nmr']:.2f}"],
            "Stillbirth Rate (per 1000)": [
                f"{maternal_indicators['stillbirth_rate']:.2f}"
            ],
            "Live Births": [maternal_indicators["live_births"]],
            "Stillbirths": [maternal_indicators["stillbirths"]],
            "Total Births": [maternal_indicators["total_births"]],
            "Low Birth Weight Rate": [
                f"{maternal_indicators['low_birth_weight_rate']:.2f}%"
            ],
        }
        newborn_df = pd.DataFrame(newborn_data_download)
        st.download_button(
            "📥 Download Newborn Data",
            data=newborn_df.to_csv(index=False),
            file_name=f"newborn_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Maternal Overview Table
    st.markdown("---")
    st.markdown("### 🤰 Maternal Care Overview")

    # Create maternal table data
    maternal_table_data = {
        "No": list(range(1, 6)),
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Mothers",
            "Total Deliveries",
            "Maternal Death Rate (per 100,000)",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_indicators['total_deliveries']:,}",
            f"{maternal_indicators['maternal_death_rate']:.2f}",
        ],
    }

    maternal_table_df = pd.DataFrame(maternal_table_data)
    st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
    st.markdown(
        maternal_table_df.style.set_table_attributes(
            'class="summary-table maternal-table"'
        )
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .to_html(),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Download button for maternal data
    col_info2, col_download2 = st.columns([3, 1])
    with col_download2:
        maternal_data_download = {
            "Start Date": [maternal_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Mothers": [maternal_tei_count],
            "Total Deliveries": [maternal_indicators["total_deliveries"]],
            "Maternal Death Rate (per 100,000)": [
                f"{maternal_indicators['maternal_death_rate']:.2f}"
            ],
        }
        maternal_df = pd.DataFrame(maternal_data_download)
        st.download_button(
            "📥 Download Maternal Data",
            data=maternal_df.to_csv(index=False),
            file_name=f"maternal_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Facility comparison table
    st.markdown("---")
    st.markdown("### 📊 Mothers & Newborns by facility")

    if facility_comparison_data:
        facilities = list(facility_comparison_data.keys())
        total_mothers = sum(
            data["mothers"] for data in facility_comparison_data.values()
        )
        total_newborns = sum(
            data["newborns"] for data in facility_comparison_data.values()
        )

        # Create transposed table structure
        transposed_data = []
        for i, facility in enumerate(facilities, 1):
            transposed_data.append(
                {
                    "No": i,
                    "Facility Name": facility,
                    "Admitted Mothers": f"{facility_comparison_data[facility]['mothers']:,}",
                    "Admitted Newborns": f"{facility_comparison_data[facility]['newborns']:,}",
                }
            )

        # Add TOTAL row
        transposed_data.append(
            {
                "No": "",
                "Facility Name": "TOTAL",
                "Admitted Mothers": f"{total_mothers:,}",
                "Admitted Newborns": f"{total_newborns:,}",
            }
        )

        transposed_df = pd.DataFrame(transposed_data)
        st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
        st.markdown(
            transposed_df.style.set_table_attributes(
                'class="summary-table newborn-table"'
            )
            .hide(axis="index")
            .set_properties(**{"text-align": "left"})
            .to_html(),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Add download button
        st.markdown("---")
        col_info3, col_download3 = st.columns([3, 1])
        with col_download3:
            download_data = []
            for facility, data in facility_comparison_data.items():
                download_data.append(
                    {
                        "Facility": data.get("full_name", facility),
                        "Admitted Mothers": data["mothers"],
                        "Admitted Newborns": data["newborns"],
                    }
                )
            download_data.append(
                {
                    "Facility": "TOTAL",
                    "Admitted Mothers": total_mothers,
                    "Admitted Newborns": total_newborns,
                }
            )
            download_df = pd.DataFrame(download_data)
            st.download_button(
                "📥 Download Facility Data",
                data=download_df.to_csv(index=False),
                file_name=f"facility_comparison_{region_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No facility data available for comparison.")


def calculate_facility_comparison_data_from_patients(
    maternal_patients, newborn_patients, facility_mapping
):
    """Optimized facility comparison data calculation using patient data"""
    facility_data = {}

    for facility_name, facility_uid in facility_mapping.items():
        # Count patients
        maternal_count = count_unique_patients(maternal_patients, [facility_uid])
        newborn_count = count_unique_patients(newborn_patients, [facility_uid])

        # Only include facilities with data
        if maternal_count > 0 or newborn_count > 0:
            short_name = shorten_facility_name(facility_name)
            facility_data[short_name] = {
                "mothers": maternal_count,
                "newborns": newborn_count,
                "full_name": facility_name,
            }

    return facility_data


def shorten_facility_name(facility_name):
    """Shorten facility name for compact display"""
    remove_words = [
        "Health Center",
        "HC",
        "Hospital",
        "Hosp",
        "Clinic",
        "Health Facility",
        "Health Post",
        "Dispensary",
        "Medical Center",
        "Medical",
        "Centre",
        "Center",
    ]

    short_name = facility_name
    for word in remove_words:
        short_name = short_name.replace(word, "").strip()

    short_name = " ".join(short_name.split())
    short_name = short_name.replace(" ,", ",").replace(",", "").strip(" -")

    # Take first 2-3 words max
    words = short_name.split()
    if len(words) > 2:
        short_name = " ".join(words[:2])

    # Final length check
    if len(short_name) > 20:
        short_name = short_name[:18] + ".."

    # If empty after processing, use first 15 chars of original
    if not short_name:
        short_name = (
            facility_name[:15] + ".." if len(facility_name) > 15 else facility_name
        )

    return short_name


def render_maternal_dashboard_shared(
    user,
    maternal_data,
    region_name,
    selected_facilities,
    facility_uids,  # ✅ This should be UIDs
    view_mode="Normal Trend",
    facility_mapping=None,
    facility_names=None,
):
    """Optimized Maternal Dashboard rendering using patient-level data with UID filtering"""

    print(f"\n{'='*80}")
    print(f"🚨 START render_maternal_dashboard_shared - DEBUG")
    print(f"{'='*80}")

    # ✅ FIXED: Only run if this is the active tab
    if st.session_state.active_tab != "maternal":
        return

    logging.info("🔄 Maternal dashboard rendering with patient-level data")

    if not maternal_data:
        st.error("No maternal data available")
        return

    # ✅ GET PATIENT-LEVEL DATA
    patients_df = maternal_data.get("patients", pd.DataFrame())

    # ✅ DEBUG: Check data structure
    print(f"\n📊 DATA STRUCTURE CHECK:")
    print(f"  Total patients in data: {len(patients_df)}")
    print(
        f"  Columns containing 'org': {[col for col in patients_df.columns if 'org' in col.lower()]}"
    )

    if "orgUnit" in patients_df.columns:
        print(f"  orgUnit column found (UIDs)")
        print(f"  Sample UIDs: {patients_df['orgUnit'].unique()[:5]}")
        print(f"  Unique orgUnits: {patients_df['orgUnit'].nunique()}")
    else:
        print(f"❌ ERROR: 'orgUnit' column not found in data!")
        st.error("Missing 'orgUnit' column in data. Cannot filter by facility UIDs.")
        return

    if "orgUnit_name" in patients_df.columns:
        print(f"  orgUnit_name column found (Names)")
        print(f"  Sample names: {patients_df['orgUnit_name'].unique()[:5]}")

    # ✅ VALIDATE UIDs are being passed correctly
    print(f"\n🔧 PARAMETER VALIDATION:")
    print(f"  Selected facilities (names): {selected_facilities}")
    print(f"  Facility UIDs received: {facility_uids}")
    print(f"  Facility names: {facility_names}")
    print(f"  View mode: {view_mode}")

    # Use patient data directly
    working_df = patients_df.copy()

    # ✅ CRITICAL FIX: Filter by UID EARLY
    if facility_uids and "orgUnit" in working_df.columns:
        before_filter = len(working_df)
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()
        after_filter = len(working_df)
        print(f"\n🔍 FILTERING BY UIDs:")
        print(f"  Before UID filter: {before_filter} patients")
        print(f"  After UID filter: {after_filter} patients")
        print(f"  Removed: {before_filter - after_filter} patients")

        # Check which UIDs actually exist in data
        found_uids = [
            uid for uid in facility_uids if uid in working_df["orgUnit"].values
        ]
        missing_uids = [
            uid for uid in facility_uids if uid not in working_df["orgUnit"].values
        ]
        print(f"  UIDs found in data: {len(found_uids)}/{len(facility_uids)}")
        if missing_uids:
            print(f"  UIDs NOT in data: {missing_uids}")

    # ✅ Validate patient counts with UIDs
    print(f"\n🔍 PATIENT COUNT VALIDATION (USING UIDs):")
    print(f"  Working dataframe: {len(working_df)} rows")
    if "tei_id" in working_df.columns:
        print(f"  Unique TEI IDs: {working_df['tei_id'].nunique()}")
    if "orgUnit" in working_df.columns:
        print(f"  Unique orgUnits: {working_df['orgUnit'].nunique()}")
        # Show breakdown by UID
        for uid in facility_uids:
            uid_patients = working_df[working_df["orgUnit"] == uid]
            uid_name = (
                patients_df[patients_df["orgUnit"] == uid]["orgUnit_name"].iloc[0]
                if not patients_df[patients_df["orgUnit"] == uid].empty
                else "Unknown"
            )
            print(
                f"  UID {uid} ({uid_name}): {len(uid_patients)} patients, {uid_patients['tei_id'].nunique()} unique TEI IDs"
            )

    # ✅ CRITICAL FIX: Ensure ALL patients have a date for filtering
    if "event_date" not in working_df.columns:
        # Try to find event date columns
        event_date_cols = [
            col for col in working_df.columns if "event_date" in col.lower()
        ]
        if event_date_cols:
            # Use the first event date column
            working_df["event_date"] = pd.to_datetime(
                working_df[event_date_cols[0]], errors="coerce"
            )
            print(f"  Using date column: {event_date_cols[0]}")
        else:
            working_df["event_date"] = pd.NaT
            print(f"⚠️ No event date columns found")

    # Ensure enrollment_date exists and is datetime
    if "enrollment_date" in working_df.columns:
        working_df["enrollment_date"] = pd.to_datetime(
            working_df["enrollment_date"], errors="coerce"
        )
    else:
        working_df["enrollment_date"] = working_df["event_date"]

    # ✅ CRITICAL: Create a combined date that always has a value
    working_df["combined_date"] = working_df["event_date"].combine_first(
        working_df["enrollment_date"]
    )

    # Check how many dates we have
    total_patients = len(working_df)
    valid_dates = working_df["combined_date"].notna().sum()

    print(f"\n📊 DATE ANALYSIS:")
    print(f"   Total patients: {total_patients}")
    print(f"   Patients with valid combined_date: {valid_dates}")

    # Store the original df for KPI calculations
    st.session_state.maternal_patients_df = working_df.copy()

    render_connection_status(working_df, user=user)

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = facility_names or selected_facilities
    st.session_state.current_comparison_mode = "facility"

    # Optimized header rendering
    if selected_facilities == ["All Facilities"]:
        header_title = f"🤰 Maternal Inpatient Data - {region_name}"
        header_subtitle = f"all {len(facility_mapping)} facilities"
    elif len(selected_facilities) == 1:
        header_title = f"🤰 Maternal Inpatient Data - {selected_facilities[0]}"
        header_subtitle = "1 facility"
    else:
        header_title = "🤰 Maternal Inpatient Data - Multiple Facilities"
        header_subtitle = f"{len(selected_facilities)} facilities"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**📊 Displaying data from {header_subtitle}**")

    # ✅ IMPROVED: Single progress container
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("---")
        st.markdown("### 📈 Preparing Dashboard...")
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.markdown(
                """
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">🔄 Processing Data</h4>
            <p style="margin: 5px 0; font-size: 14px;">• Computing KPIs and indicators...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Generating charts and visualizations...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Preparing data tables...</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">⏱️ This may take 2-4 minutes depending on data size</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with progress_col2:
            st.markdown(
                """
            <div style="text-align: center; padding: 10px;">
            <div style="font-size: 24px;">⏳</div>
            <div style="font-size: 12px; margin-top: 5px;">Processing</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Create containers for better performance
    kpi_container = st.container()

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            working_df, container=col_ctrl, context="regional_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ✅ CRITICAL FIX: Use THE SAME filtered data for BOTH KPIs and charts
    # Apply date filters FIRST to get the correct time period
    print(f"\n🔍 APPLYING DATE FILTERS...")
    filtered_for_all = apply_simple_filters(working_df, filters, facility_uids)

    print(f"\n📊 FILTERING ANALYSIS:")
    print(f"   Original patients (after UID filter): {len(working_df)}")
    print(f"   After date filtering: {len(filtered_for_all)}")

    if "period_display" in filtered_for_all.columns:
        unique_periods = filtered_for_all["period_display"].unique()
        print(f"   Periods in filtered data: {list(unique_periods)}")

    # Store BOTH versions - but use filtered_for_all for everything
    st.session_state["filtered_patients"] = filtered_for_all.copy()
    st.session_state["all_patients_for_kpi"] = (
        filtered_for_all.copy()
    )  # ✅ FIX: Use FILTERED data

    # KPI Cards with FILTERED data (respects time period)
    with kpi_container:
        location_name, location_type = get_location_display_name(
            selected_facilities, region_name
        )

        user_id = str(user.get("id", user.get("username", "default_user")))

        # ✅ FIX: Use FILTERED data for KPI calculations
        print(f"\n📊 KPI CALCULATION (AFTER FIX):")
        print(f"   Using {len(filtered_for_all)} patients for KPI calculations")

        kpi_data = render_kpi_cards(filtered_for_all, location_name, user_id=user_id)

        # Save for summary dashboard to reuse
        st.session_state.last_computed_kpis = kpi_data
        st.session_state.last_computed_facilities = facility_uids
        st.session_state.last_computed_timestamp = time.time()

    # ✅ CLEAR THE PROGRESS INDICATOR ONCE KPI CARDS ARE DONE
    progress_container.empty()

    # Charts section with optimized rendering
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        selected_kpi = render_kpi_tab_navigation()

        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - Facility Comparison - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            render_comparison_chart(
                kpi_selection=selected_kpi,
                patient_df=filtered_for_all,  # ✅ Use same filtered data
                comparison_mode="facility",
                display_names=facility_names or selected_facilities,
                facility_uids=facility_uids,  # ✅ Pass UIDs
                facilities_by_region=None,
                region_names=None,
                bg_color=bg_color,
                text_color=text_color,
                is_national=False,
                show_table=filters.get("show_table", False),
            )
        else:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            render_trend_chart_section(
                selected_kpi,
                filtered_for_all,  # ✅ Use same filtered data
                facility_uids,  # ✅ Pass UIDs
                facility_names or selected_facilities,
                bg_color,
                text_color,
                comparison_mode="facility",
                facilities_by_region=None,
                region_names=None,
            )

        render_additional_analytics(
            selected_kpi,
            filtered_for_all,  # ✅ Use same filtered data
            facility_uids,  # ✅ Pass UIDs
            bg_color,
            text_color,
        )


def debug_facility_spacing_issues(maternal_data, facilities_from_db, facility_mapping):
    """
    Debug function to identify facility name spacing and formatting issues
    """
    print(f"\n{'='*100}")
    print(f"🔍 DEBUG: FACILITY SPACING & FORMATTING ISSUES")
    print(f"{'='*100}")

    if not maternal_data:
        print("❌ No maternal data available")
        return

    patients_df = maternal_data.get("patients", pd.DataFrame())

    if patients_df.empty:
        print("❌ Empty patients dataframe")
        return

    # Get facilities from CSV data
    if "orgUnit_name" not in patients_df.columns:
        print("❌ 'orgUnit_name' column not found in patient data")
        return

    csv_facilities = patients_df["orgUnit_name"].unique()
    db_facility_names = [f[0] for f in facilities_from_db]

    print(f"\n📊 OVERVIEW:")
    print(f"   CSV facilities: {len(csv_facilities)}")
    print(f"   DB facilities: {len(db_facility_names)}")
    print(f"   Facility mapping entries: {len(facility_mapping)}")

    # Create a function to normalize facility names for comparison
    def normalize_facility_name(name):
        """Normalize facility name by removing extra spaces and standardizing"""
        if not name or pd.isna(name):
            return ""

        name = str(name).strip()

        # Remove multiple spaces
        import re

        name = re.sub(r"\s+", " ", name)

        # Standardize hospital types (lowercase for comparison)
        name_lower = name.lower()

        # Standardize common terms
        replacements = [
            ("health center", "hc"),
            ("primary hospital", "ph"),
            ("general hospital", "gh"),
            ("specialized hospital", "sh"),
            ("referral hospital", "rh"),
            ("medical center", "mc"),
            ("clinic", "cl"),
            ("health post", "hp"),
            ("dispensary", "disp"),
        ]

        for old, new in replacements:
            if old in name_lower:
                name_lower = name_lower.replace(old, new)

        # Remove all spaces and punctuation for strict matching
        name_strict = re.sub(r"[^a-z0-9]", "", name_lower)

        return {"original": name, "normalized": name_lower, "strict": name_strict}

    # Create normalized dictionaries
    print(f"\n🔍 CREATING NORMALIZED DICTIONARIES...")

    csv_normalized = {}
    for fac in csv_facilities:
        norm = normalize_facility_name(fac)
        csv_normalized[norm["strict"]] = {
            "original": norm["original"],
            "normalized": norm["normalized"],
            "strict": norm["strict"],
            "patient_count": patients_df[patients_df["orgUnit_name"] == fac][
                "tei_id"
            ].nunique(),
        }

    db_normalized = {}
    for fac in db_facility_names:
        norm = normalize_facility_name(fac)
        db_normalized[norm["strict"]] = {
            "original": norm["original"],
            "normalized": norm["normalized"],
            "strict": norm["strict"],
        }

    mapping_normalized = {}
    for fac in facility_mapping.keys():
        norm = normalize_facility_name(fac)
        mapping_normalized[norm["strict"]] = {
            "original": norm["original"],
            "normalized": norm["normalized"],
            "strict": norm["strict"],
        }

    # Find matches and mismatches
    print(f"\n📊 MATCHING ANALYSIS:")

    # CSV vs DB
    csv_in_db = set(csv_normalized.keys()) & set(db_normalized.keys())
    csv_not_in_db = set(csv_normalized.keys()) - set(db_normalized.keys())
    db_not_in_csv = set(db_normalized.keys()) - set(csv_normalized.keys())

    print(f"   CSV facilities matched with DB: {len(csv_in_db)}/{len(csv_normalized)}")
    print(f"   CSV facilities NOT in DB: {len(csv_not_in_db)}")
    print(f"   DB facilities NOT in CSV: {len(db_not_in_csv)}")

    # CSV vs Mapping
    csv_in_mapping = set(csv_normalized.keys()) & set(mapping_normalized.keys())
    csv_not_in_mapping = set(csv_normalized.keys()) - set(mapping_normalized.keys())

    print(
        f"\n   CSV facilities in mapping: {len(csv_in_mapping)}/{len(csv_normalized)}"
    )
    print(f"   CSV facilities NOT in mapping: {len(csv_not_in_mapping)}")

    # Show specific mismatches
    print(f"\n🔍 SPECIFIC MISMATCHES - CSV NOT IN MAPPING:")
    if csv_not_in_mapping:
        print(f"   Found {len(csv_not_in_mapping)} CSV facilities not in mapping")

        # Sort by patient count (highest first)
        mismatches_with_counts = []
        for strict_name in csv_not_in_mapping:
            csv_info = csv_normalized[strict_name]
            mismatches_with_counts.append(
                {
                    "strict": strict_name,
                    "original": csv_info["original"],
                    "patient_count": csv_info["patient_count"],
                    "normalized": csv_info["normalized"],
                }
            )

        # Sort by patient count descending
        mismatches_with_counts.sort(key=lambda x: x["patient_count"], reverse=True)

        print(f"\n   Top 20 mismatches by patient count:")
        for i, mismatch in enumerate(mismatches_with_counts[:20]):
            print(f"   {i+1:2d}. '{mismatch['original']}'")
            print(f"       Patients: {mismatch['patient_count']}")
            print(f"       Normalized: {mismatch['normalized']}")
            print(f"       Strict key: {mismatch['strict']}")

            # Try to find similar names in mapping
            similar_in_mapping = []
            for map_strict, map_info in mapping_normalized.items():
                # Check for partial matches
                if (
                    mismatch["strict"] in map_strict
                    or map_strict in mismatch["strict"]
                    or mismatch["normalized"].replace(" ", "")
                    in map_info["normalized"].replace(" ", "")
                    or map_info["normalized"].replace(" ", "")
                    in mismatch["normalized"].replace(" ", "")
                ):
                    similar_in_mapping.append(map_info["original"])

            if similar_in_mapping:
                print(f"       Similar in mapping: {similar_in_mapping}")
            print()

    # Show facilities with many patients but no mapping
    print(f"\n🔍 HIGH-IMPACT MISMATCHES (Many patients, no mapping):")
    high_impact = [m for m in mismatches_with_counts if m["patient_count"] > 0]
    if high_impact:
        print(f"   Found {len(high_impact)} facilities with patients but no mapping")
        for i, mismatch in enumerate(high_impact[:10]):
            print(
                f"   {i+1}. '{mismatch['original']}': {mismatch['patient_count']} patients"
            )

    # Show exact string differences for sample mismatches
    print(f"\n🔍 STRING COMPARISON FOR SAMPLE MISMATCHES:")
    sample_count = min(5, len(csv_not_in_mapping))
    for i, strict_name in enumerate(list(csv_not_in_mapping)[:sample_count]):
        csv_info = csv_normalized[strict_name]
        print(f"\n   CSV facility {i+1}:")
        print(f"     Original: '{csv_info['original']}'")
        print(f"     Length: {len(csv_info['original'])} chars")
        print(f"     ASCII codes: {[ord(c) for c in csv_info['original']]}")

        # Show character by character
        print(f"     Characters:")
        for j, char in enumerate(csv_info["original"]):
            print(f"       [{j:2d}] '{char}' (ASCII: {ord(char):3d})")

    # Check for common patterns
    print(f"\n🔍 COMMON PATTERN ANALYSIS:")

    # Check for trailing/leading spaces
    csv_with_spaces = []
    for fac in csv_facilities:
        if str(fac) != str(fac).strip():
            csv_with_spaces.append(fac)

    if csv_with_spaces:
        print(f"   CSV facilities with extra spaces: {len(csv_with_spaces)}")
        for fac in csv_with_spaces[:5]:
            print(f"     '{fac}' (stripped: '{fac.strip()}')")

    # Check for different capitalization
    print(f"\n   CSV facility name variations:")
    name_variations = {}
    for fac in csv_facilities:
        key = str(fac).lower().strip()
        if key not in name_variations:
            name_variations[key] = []
        if str(fac) not in name_variations[key]:
            name_variations[key].append(str(fac))

    variations_with_multiple = {k: v for k, v in name_variations.items() if len(v) > 1}
    if variations_with_multiple:
        print(
            f"   Found {len(variations_with_multiple)} names with multiple variations:"
        )
        for key, variations in list(variations_with_multiple.items())[:5]:
            print(f"     '{key}':")
            for var in variations:
                print(f"       - '{var}'")

    # Quick fix suggestions
    print(f"\n🔧 QUICK FIX SUGGESTIONS:")

    # 1. Create a manual mapping for mismatched facilities
    print(f"\n   1. MANUAL MAPPING SCRIPT (add to your code):")
    print(f"   manual_facility_fixes = {{")

    for i, mismatch in enumerate(mismatches_with_counts[:10]):
        if mismatch["patient_count"] > 0:
            csv_name = mismatch["original"]
            # Find the closest match in DB
            closest_match = None
            closest_score = 0

            for db_fac in db_facility_names:
                # Simple similarity check
                csv_lower = csv_name.lower().replace(" ", "")
                db_lower = db_fac.lower().replace(" ", "")

                if csv_lower in db_lower or db_lower in csv_lower:
                    similarity = len(set(csv_lower) & set(db_lower)) / max(
                        len(csv_lower), len(db_lower)
                    )
                    if similarity > closest_score:
                        closest_score = similarity
                        closest_match = db_fac

            if closest_match and closest_score > 0.7:
                print(
                    f"       '{csv_name}': '{closest_match}',  # {mismatch['patient_count']} patients"
                )

    print(f"   }}")

    # 2. Auto-clean function
    print(f"\n   2. AUTO-CLEAN FUNCTION:")
    print(f"   def clean_facility_name(name):")
    print(f"       import re")
    print(f"       name = str(name).strip()")
    print(f"       name = re.sub(r'\\s+', ' ', name)  # Remove extra spaces")
    print(f"       # Common fixes")
    print(f"       fixes = {{")
    print(f"           'Abiadi  General hospital': 'Abiadi General hospital',")
    print(f"           'Abiadi General  hospital': 'Abiadi General hospital',")
    print(f"       }}")
    print(f"       return fixes.get(name, name)")

    print(f"\n{'='*100}")
    print(f"🔍 DEBUG: END FACILITY SPACING ANALYSIS")
    print(f"{'='*100}\n")


def diagnose_data_issues(user, shared_data, selected_facilities, facility_mapping):
    """Diagnostic function to identify data issues"""
    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 Run Data Diagnostics", use_container_width=True):
        with st.expander("📊 Data Diagnostics Report", expanded=True):
            st.markdown("### 🩺 Data Health Check")

            # Get data
            maternal_data = shared_data.get("maternal", {})
            newborn_data = shared_data.get("newborn", {})

            maternal_patients = maternal_data.get("patients", pd.DataFrame())
            newborn_patients = newborn_data.get("patients", pd.DataFrame())

            # Facility analysis
            st.markdown("#### 🏥 Facility Analysis")

            # Get normalized facility names
            from utils.data_service import normalize_facility_name

            db_facilities = [f[0] for f in st.session_state.facilities]
            normalized_db = [normalize_facility_name(f) for f in db_facilities]

            if (
                "orgUnit_name" in maternal_patients.columns
                and not maternal_patients.empty
            ):
                data_facilities = maternal_patients["orgUnit_name"].unique()
                normalized_data = [normalize_facility_name(f) for f in data_facilities]

                # Find matches and mismatches
                matches = set(normalized_db) & set(normalized_data)
                db_only = set(normalized_db) - set(normalized_data)
                data_only = set(normalized_data) - set(normalized_db)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Matches", len(matches))
                with col2:
                    st.metric("❌ DB Only", len(db_only))
                with col3:
                    st.metric("❓ Data Only", len(data_only))

                if db_only:
                    st.warning("Facilities in DB but not in data:")
                    for fac in sorted(db_only)[:10]:
                        st.write(f"  - {fac}")

                if data_only:
                    st.info("Facilities in data but not in DB:")
                    for fac in sorted(data_only)[:10]:
                        st.write(f"  - {fac}")
            else:
                st.warning("No facility names found in maternal patient data")

            # Patient count analysis
            st.markdown("#### 👥 Patient Count Analysis")

            if not maternal_patients.empty and "tei_id" in maternal_patients.columns:
                total_patients = maternal_patients["tei_id"].nunique()
                st.write(f"**Total unique patients:** {total_patients}")

                # Count by facility
                if "orgUnit_name" in maternal_patients.columns:
                    facility_counts = maternal_patients.groupby("orgUnit_name")[
                        "tei_id"
                    ].nunique()

                    st.write("**Top 10 facilities by patient count:**")
                    for facility, count in (
                        facility_counts.sort_values(ascending=False).head(10).items()
                    ):
                        st.write(f"  - {facility}: {count} patients")

            # Data quality checks
            st.markdown("#### 📋 Data Quality Checks")

            if not maternal_patients.empty:
                # Check for missing critical columns
                critical_columns = [
                    "birth_outcome_delivery_summary",
                    "mode_of_delivery_maternal_delivery_summary",
                    "event_date_delivery_summary",
                ]

                missing_cols = [
                    col
                    for col in critical_columns
                    if col not in maternal_patients.columns
                ]
                if missing_cols:
                    st.error(f"Missing critical columns: {missing_cols}")

                # Check birth outcome data
                if "birth_outcome_delivery_summary" in maternal_patients.columns:
                    total_rows = len(maternal_patients)
                    non_null = (
                        maternal_patients["birth_outcome_delivery_summary"]
                        .notna()
                        .sum()
                    )

                    from utils.kpi_utils import compute_stillbirth_count

                    stillbirths = compute_stillbirth_count(maternal_patients)

                    st.write(f"**Birth Outcome Analysis:**")
                    st.write(f"  - Total rows: {total_rows}")
                    st.write(
                        f"  - Non-null birth outcomes: {non_null} ({non_null/total_rows*100:.1f}%)"
                    )
                    st.write(f"  - Stillbirths found: {stillbirths}")


def validate_patient_counts(maternal_data, facility_uids, facility_names):
    """Validate that all patients are being counted correctly using UIDs"""
    print(f"\n{'='*80}")
    print(f"🔍 PATIENT COUNT VALIDATION (USING UIDs)")
    print(f"{'='*80}")

    if not maternal_data:
        print("❌ No maternal data available")
        return

    patients_df = maternal_data.get("patients", pd.DataFrame())

    if patients_df.empty:
        print("❌ Empty patients dataframe")
        return

    print(f"Total patients in dataset: {len(patients_df)}")
    print(f"Unique TEI IDs in dataset: {patients_df['tei_id'].nunique()}")

    # Check if we have the orgUnit column (which contains UIDs)
    if "orgUnit" in patients_df.columns:
        print(f"✅ Found 'orgUnit' column (UIDs)")

        # Get unique orgUnits in data
        unique_orgunits = patients_df["orgUnit"].unique()
        print(f"Unique orgUnits in data: {len(unique_orgunits)}")
        print(
            f"Sample orgUnits: {unique_orgunits[:10] if len(unique_orgunits) > 10 else unique_orgunits}"
        )
    else:
        print("❌ 'orgUnit' column NOT FOUND - cannot filter by UID")
        return

    # Filter by selected facilities USING UIDs
    if facility_uids:
        print(f"\nFor selected facilities ({len(facility_uids)} facilities):")
        print(f"  Selected facility UIDs: {facility_uids}")

        # Find which UIDs exist in the data
        existing_uids = [
            uid for uid in facility_uids if uid in patients_df["orgUnit"].values
        ]
        missing_uids = [
            uid for uid in facility_uids if uid not in patients_df["orgUnit"].values
        ]

        print(f"  UIDs found in data: {len(existing_uids)}")
        print(f"  UIDs NOT in data: {len(missing_uids)}")

        if missing_uids:
            print(f"  Missing UIDs: {missing_uids}")

        # Filter by UIDs
        filtered_df = patients_df[patients_df["orgUnit"].isin(facility_uids)]
        print(f"\n  Filtered patients: {len(filtered_df)}")
        print(f"  Unique TEI IDs after filtering: {filtered_df['tei_id'].nunique()}")

        # Show breakdown by facility UID
        print(f"\nBreakdown by facility UID:")
        for facility_name, facility_uid in zip(facility_names, facility_uids):
            facility_patients = filtered_df[filtered_df["orgUnit"] == facility_uid]
            unique_patients = facility_patients["tei_id"].nunique()
            print(f"  {facility_name} ({facility_uid}): {unique_patients} patients")

    # Also check if we have facility names for debugging
    if "orgUnit_name" in patients_df.columns:
        print(f"\n📋 Facility names in data (first 10):")
        unique_names = patients_df["orgUnit_name"].unique()[:10]
        for name in unique_names:
            # Find the corresponding UID for this name
            sample_uid = (
                patients_df[patients_df["orgUnit_name"] == name]["orgUnit"].iloc[0]
                if not patients_df[patients_df["orgUnit_name"] == name].empty
                else "N/A"
            )
            print(f"  '{name}' → UID: {sample_uid}")

    print(f"{'='*80}\n")


def render():
    """Main optimized render function"""
    st.set_page_config(
        page_title="Regional Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Re-initialize session state for safety
    initialize_session_state()

    # Show user change notification if needed
    if st.session_state.get("user_changed", False):
        st.sidebar.info("👤 User changed - loading fresh data...")
        current_user = st.session_state.get("user", {})
        clear_shared_cache(current_user)

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

    try:
        with open("utils/national.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Get user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    region_name = user.get("region_name", "Unknown Region")

    # Compact sidebar user info
    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 {username}</div>
            <div>🗺️ {region_name}</div>
            <div>🛡️ {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh Data Button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        clear_shared_cache(user)
        st.session_state.refresh_trigger = True  # ✅ Set refresh trigger
        st.session_state.selection_applied = True
        st.session_state.facility_filter_applied = False
        st.session_state.last_computed_kpis = None
        st.session_state.last_computed_newborn_kpis = None
        st.session_state.summary_kpi_cache = {}
        st.session_state.data_initialized = False

        # Clear data signature to force fresh computation
        current_user_key = (
            f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        )
        data_changed_key = f"data_signature_{current_user_key}"
        if data_changed_key in st.session_state:
            del st.session_state[data_changed_key]

        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
        st.rerun()

    # ================ OPTIMIZED DATA LOADING ================
    # Get static data (cached for 1 hour)
    if not st.session_state.get("static_data_loaded", False) or st.session_state.get(
        "user_changed", False
    ):
        with st.sidebar:
            with st.spinner("🚀 Loading facility data..."):
                static_data = get_static_data(user)
                st.session_state.facilities = static_data["facilities"]
                st.session_state.facility_mapping = static_data["facility_mapping"]
                st.session_state.program_uid_map = static_data["program_uid_map"]
                st.session_state.static_data_loaded = True
                st.session_state.user_changed = False

    facilities = st.session_state.facilities
    facility_mapping = st.session_state.facility_mapping
    program_uid_map = st.session_state.program_uid_map

    # ✅ FIX: SINGLE DATA LOADING - Only load once and store in variable
    if not st.session_state.get("data_initialized", False):
        # First time or fresh data needed
        with st.spinner("🚀 Loading dashboard data..."):
            shared_data = get_shared_program_data_optimized(
                user, program_uid_map, show_spinner=False
            )
            st.session_state.data_initialized = True
            st.session_state.cached_shared_data = (
                shared_data  # ✅ Store in session state
            )
            logging.info("✅ Initial data loading complete")
    else:
        # Use cached data from session state - no loading needed
        shared_data = st.session_state.cached_shared_data
        logging.info("✅ Using cached shared data from session state")

    # ✅ Add cache status indicator in sidebar
    if st.session_state.get("data_initialized", False):
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        timestamp_key = f"shared_data_timestamp_{user_key}"

        if timestamp_key in st.session_state:
            time_elapsed = time.time() - st.session_state[timestamp_key]
            minutes_old = int(time_elapsed // 60)
            seconds_old = int(time_elapsed % 60)

            if minutes_old < 30:
                st.sidebar.info(f"🔄 Data: {minutes_old}m {seconds_old}s old")
            else:
                st.sidebar.warning(f"🔄 Data: {minutes_old}m old (will auto-refresh)")

    # ================ COMPACT FACILITY SELECTION ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">🏥 Facility Selection</p>',
        unsafe_allow_html=True,
    )

    # Get current selection with safe default
    current_selection = st.session_state.get("selected_facilities", ["All Facilities"])

    # Facility selection with optimized layout
    with st.sidebar.form("facility_selection_form", border=False):
        # Simple multiselect for facilities
        facility_options = ["All Facilities"] + [f[0] for f in facilities]

        selected_facilities = st.multiselect(
            "Choose facilities:",
            options=facility_options,
            default=current_selection,
            help="Select facilities to display data for",
            key="facility_multiselect",
            label_visibility="collapsed",
        )

        # Handle "All Facilities" logic
        if "All Facilities" in selected_facilities:
            if len(selected_facilities) > 1:
                # Remove "All Facilities" if other facilities are selected
                selected_facilities = [
                    f for f in selected_facilities if f != "All Facilities"
                ]
            else:
                # Only "All Facilities" is selected
                selected_facilities = ["All Facilities"]

        selection_submitted = st.form_submit_button(
            "✅ Apply Selection", use_container_width=True
        )
        if selection_submitted:
            st.session_state.selected_facilities = selected_facilities
            st.session_state.selection_applied = True
            st.session_state.facility_filter_applied = True
            st.rerun()

    # Display selection summary
    total_facilities = len(facilities)
    selected_facilities = st.session_state.get(
        "selected_facilities", ["All Facilities"]
    )
    if selected_facilities == ["All Facilities"]:
        display_text = f"Selected: All ({total_facilities})"
    else:
        display_text = f"Selected: {len(selected_facilities)} / {total_facilities}"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: -10px;'>{display_text}</p>",
        unsafe_allow_html=True,
    )

    # Get facility UIDs for selected facilities
    if selected_facilities == ["All Facilities"]:
        facility_uids = list(facility_mapping.values())
        facility_names = ["All Facilities"]
    else:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        facility_names = selected_facilities

    # ================ COMPACT VIEW MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">📊 View Mode</p>',
        unsafe_allow_html=True,
    )

    view_mode = "Normal Trend"
    if selected_facilities != ["All Facilities"] and len(selected_facilities) > 1:
        view_mode = st.sidebar.radio(
            "View:",
            ["Normal Trend", "Facility Comparison"],
            index=0,
            key="view_mode_regional",
            label_visibility="collapsed",
        )

    # ================ DIAGNOSTIC TOOL ================
    diagnose_data_issues(user, shared_data, selected_facilities, facility_mapping)

    # ================ OPTIMIZED TABS WITH PROPER ISOLATION ================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🤰 **Maternal**",
            "👶 **Newborn**",
            "📊 **Summary**",
            "📋 **Mentorship**",
            "🔍 **Data Quality**",
        ]
    )

    with tab1:
        # Update active tab only if this tab is clicked
        if st.session_state.active_tab != "maternal":
            st.session_state.active_tab = "maternal"
            logging.info("🔄 Switched to Maternal tab")

        maternal_data = shared_data["maternal"]
        if maternal_data:
            render_maternal_dashboard_shared(
                user,
                maternal_data,
                region_name,
                selected_facilities,
                facility_uids,
                view_mode=view_mode,
                facility_mapping=facility_mapping,
                facility_names=facility_names,
            )
        else:
            st.error("Maternal data not available")

    with tab2:
        if st.session_state.active_tab != "newborn":
            st.session_state.active_tab = "newborn"
            logging.info("🔄 Switched to Newborn tab")

        newborn_data = shared_data["newborn"]
        if newborn_data:
            # Note: You'll need to update render_newborn_dashboard similarly
            # For now, let it use the transformed events data
            render_newborn_dashboard(
                user=user,
                program_uid=program_uid_map.get("Newborn Care Form"),
                region_name=region_name,
                selected_facilities=selected_facilities,
                facility_uids=facility_uids,
                view_mode=view_mode,
                facility_mapping=facility_mapping,
                facility_names=facility_names,
                shared_newborn_data=newborn_data,
            )
        else:
            st.error("Newborn data not available")

    with tab3:
        if st.session_state.active_tab != "summary":
            st.session_state.active_tab = "summary"
            logging.info("🔄 Switched to Summary tab")

        render_summary_dashboard_shared(
            user,
            region_name,
            facility_mapping,
            selected_facilities,
            shared_data,
        )

    with tab4:
        if st.session_state.active_tab != "mentorship":
            st.session_state.active_tab = "mentorship"
            logging.info("🔄 Switched to Mentorship tab")

        # ✅ NEW: Check if mentorship data should be loaded
        if not st.session_state.tab_data_loaded["mentorship"]:
            render_tab_placeholder(
                "Mentorship Dashboard",
                "📋",
                "mentorship",
                "View mentorship tracking data and ODK form submissions",
            )
        else:
            # ✅ Show loading indicator if data is being processed
            if st.session_state.tab_loading["mentorship"]:
                render_loading_indicator("Mentorship Dashboard", "📋")
                # After loading, set loading to False
                st.session_state.tab_loading["mentorship"] = False
                st.rerun()
            display_odk_dashboard(user)

    with tab5:
        if st.session_state.active_tab != "data_quality":
            st.session_state.active_tab = "data_quality"
            logging.info("🔄 Switched to Data Quality tab")

        render_data_quality_tracking(user)

    # Log current active tab state
    logging.info(f"📊 Current active tab: {st.session_state.active_tab}")

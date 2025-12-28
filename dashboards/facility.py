# dashboards/facility.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from utils.kpi_utils import clear_cache
from components.kpi_card import render_kpi_cards
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


def calculate_maternal_indicators_from_patients(patient_df, facility_uid):
    """Calculate maternal indicators from patient-level data"""
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

    # Filter by facility if specified
    if facility_uid and "orgUnit" in patient_df.columns:
        filtered_df = patient_df[patient_df["orgUnit"] == facility_uid]
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


def calculate_newborn_indicators_from_patients(patient_df, facility_uid):
    """Calculate newborn indicators from patient-level data"""
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

    # Filter by facility if specified
    if facility_uid and "orgUnit" in patient_df.columns:
        filtered_df = patient_df[patient_df["orgUnit"] == facility_uid]
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


def get_location_display_name_facility(facility_name):
    """Get location display name for facility level"""
    return facility_name, "Facility"


def render_summary_dashboard_facility(user, facility_name, facility_uid, shared_data):
    """Optimized Summary Dashboard for facility using patient-level data"""

    # Get location display name
    location_name, location_type = get_location_display_name_facility(facility_name)

    st.markdown(
        f'<div class="main-header">📊 Summary Dashboard - {location_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Comprehensive overview of maternal and newborn health indicators**")

    # Use shared data instead of fetching separately
    maternal_data = shared_data["maternal"]
    newborn_data = shared_data["newborn"]

    # Extract PATIENT-LEVEL dataframes
    maternal_patients = (
        maternal_data.get("patients", pd.DataFrame())
        if maternal_data
        else pd.DataFrame()
    )
    newborn_patients = (
        newborn_data.get("patients", pd.DataFrame()) if newborn_data else pd.DataFrame()
    )

    # Create cache key for summary data
    cache_key = f"summary_facility_{facility_name}_{facility_uid}_{len(maternal_patients)}_{len(newborn_patients)}"

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
            maternal_indicators = calculate_maternal_indicators_from_patients(
                maternal_patients, facility_uid
            )
            newborn_indicators = calculate_newborn_indicators_from_patients(
                newborn_patients, facility_uid
            )

            # Get patient counts
            maternal_patient_count = count_unique_patients(
                maternal_patients, facility_uid
            )
            newborn_patient_count = count_unique_patients(
                newborn_patients, facility_uid
            )

            # Get earliest dates
            maternal_start_date = get_earliest_date(
                maternal_patients, "enrollment_date"
            )
            newborn_start_date = get_earliest_date(newborn_patients, "enrollment_date")

            summary_data = {
                "maternal_indicators": maternal_indicators,
                "newborn_indicators": newborn_indicators,
                "maternal_tei_count": maternal_patient_count,
                "newborn_tei_count": newborn_patient_count,
                "newborn_start_date": newborn_start_date,
                "maternal_start_date": maternal_start_date,
            }

            # Cache the computed data
            st.session_state.summary_kpi_cache_facility[cache_key] = {
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

    # Show quick statistics at the top
    st.markdown("### 📈 Quick Statistics")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (
            col1,
            "Total Mothers",
            maternal_tei_count,
            "Unique mothers admitted",
        ),
        (
            col2,
            "Total Newborns",
            newborn_tei_count,
            "Unique newborns admitted",
        ),
        (
            col3,
            "Maternal Deaths",
            maternal_indicators["maternal_deaths"],
            "Maternal mortality cases",
        ),
        (
            col4,
            "Coverage Start",
            maternal_start_date,
            "Earliest data record",
        ),
    ]

    for col, label, value, help_text in metrics:
        with col:
            st.markdown(
                f"""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1f77b4; margin-bottom: 10px;">
                <div style="font-size: 0.8rem; color: #666; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4; margin: 10px 0;">{value if isinstance(value, int) else value}</div>
                <div style="font-size: 0.65rem; color: #888;">{help_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Newborn Overview Table
    st.markdown("### 👶 Newborn Care Overview")

    # Create newborn table data
    newborn_table_data = {
        "No": list(range(1, 7)),
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Newborns",
            "NMR (per 1000)",
            "KMC Coverage (%)",
            "Low Birth Weight Count (<2500g)",
        ],
        "Value": [
            newborn_start_date,
            location_name,
            f"{newborn_tei_count:,}",
            f"{newborn_indicators['nmr']:.2f}",
            f"{newborn_indicators['kmc_coverage_rate']:.2f}%",
            f"{newborn_indicators['total_lbw']:,}",
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

    # Download button for newborn data
    col_info, col_download = st.columns([3, 1])
    with col_download:
        newborn_data_download = {
            "Start Date": [newborn_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Newborns": [newborn_tei_count],
            "NMR (per 1000)": [f"{newborn_indicators['nmr']:.2f}"],
            "KMC Coverage (%)": [f"{newborn_indicators['kmc_coverage_rate']:.2f}"],
            "Low Birth Weight Count": [newborn_indicators["total_lbw"]],
        }
        newborn_df = pd.DataFrame(newborn_data_download)
        st.download_button(
            "Download Newborn Data",
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
        "No": list(range(1, 9)),
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Mothers",
            "Total Deliveries",
            "Maternal Death Rate (per 100,000)",
            "Live Births",
            "Stillbirths",
            "Stillbirth Rate (per 1000)",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_indicators['total_deliveries']:,}",
            f"{maternal_indicators['maternal_death_rate']:.2f}",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['stillbirth_rate']:.2f}",
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
            "Live Births": [maternal_indicators["live_births"]],
            "Stillbirths": [maternal_indicators["stillbirths"]],
            "Stillbirth Rate (per 1000)": [
                f"{maternal_indicators['stillbirth_rate']:.2f}"
            ],
        }
        maternal_df = pd.DataFrame(maternal_data_download)
        st.download_button(
            "Download Maternal Data",
            data=maternal_df.to_csv(index=False),
            file_name=f"maternal_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
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

    # KPI Cards with FILTERED data
    with kpi_container:
        location_name, location_type = get_location_display_name_facility(facility_name)

        user_id = str(user.get("id", user.get("username", "default_user")))
        kpi_data = render_kpi_cards(filtered_for_all, location_name, user_id=user_id)

        # Save for summary dashboard to reuse
        st.session_state.last_computed_kpis = kpi_data
        st.session_state.last_computed_facilities = (
            [facility_uid] if facility_uid else []
        )
        st.session_state.last_computed_timestamp = time.time()

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

        newborn_data = shared_data["newborn"]
        if newborn_data:
            # Pass shared data to newborn dashboard
            from newborns_dashboard.facility_newborn import render_newborn_dashboard

            render_newborn_dashboard(
                user=user,
                program_uid=program_uid_map.get("Newborn Care Form"),
                facility_name=facility_name,
                facility_uid=facility_uid,
                view_mode=view_mode,
                shared_newborn_data=newborn_data,
            )
        else:
            st.error("Newborn data not available")

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

# dashboards/national.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from components.kpi_card import render_kpi_cards
from newborns_dashboard.national_newborn import render_newborn_dashboard
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_grouped_by_region,
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
)
from utils.kpi_utils import clear_cache, compute_kpis
from utils.odk_dashboard import display_odk_dashboard
from dashboards.data_quality_tracking import render_data_quality_tracking
from newborns_dashboard.kpi_nmr import compute_nmr_kpi


logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# Performance optimization: Pre-load essential data with user-specific caching
@st.cache_data(ttl=3600, show_spinner=False)
def get_static_data(user):
    """Cache static data that doesn't change often - user-specific"""
    user_identifier = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"

    facilities_by_region = get_facilities_grouped_by_region(user)
    facility_mapping = get_facility_mapping_for_user(user)
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    return {
        "facilities_by_region": facilities_by_region,
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

    # Check if cache has expired (30 minutes = 1800 seconds)
    cache_expired = False
    if shared_timestamp_key in st.session_state:
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        cache_expired = time_elapsed > 1800
        if cache_expired:
            logging.info(
                f"Cache expired after {time_elapsed:.0f} seconds, fetching fresh data"
            )

    # Check if user changed (force fresh data)
    user_changed = st.session_state.get("user_changed", False)

    # Determine if we need fresh data
    need_fresh_data = (
        not st.session_state.get(shared_loaded_key, False)
        or cache_expired
        or user_changed
        or st.session_state.get("refresh_trigger", False)
    )

    if need_fresh_data:
        logging.info(
            "Fetching fresh data (cache expired, user changed, or manual refresh)"
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
                        f"Fresh data: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients"
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
            f"Using cached data: {maternal_patient_count} maternal patients, {newborn_patient_count} newborn patients ({time_elapsed:.0f}s old)"
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

        logging.info("Cleared user-specific cache")
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
        logging.info("Cleared ALL shared caches")

    clear_cache()


def initialize_session_state():
    """Optimized session state initialization with proper tab isolation"""
    session_vars = {
        "refresh_trigger": False,
        "selected_facilities": ["All Facilities"],
        "selected_regions": [],
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
        "filter_mode": "All Facilities",
        "filtered_patients": pd.DataFrame(),
        "selection_applied": True,
        "cached_patients_data": None,
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
        "last_computed_kpis": None,
        "last_computed_facilities": None,
        "last_computed_timestamp": None,
        "last_computed_newborn_kpis": None,
        "last_computed_newborn_timestamp": None,
        "summary_kpi_cache": {},
        "active_tab": "maternal",
        "data_initialized": False,
        "tab_initialized": {
            "maternal": False,
            "newborn": False,
            "summary": False,
            "mentorship": False,
            "data_quality": False,
        },
        "tab_data_loaded": {
            "maternal": True,
            "newborn": True,
            "summary": False,
            "mentorship": False,
            "data_quality": True,
        },
        "tab_loading": {
            "summary": False,
            "mentorship": False,
        },
        "show_summarized_data": True,
        "facilities_by_region": {},
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
        st.session_state.static_data_loaded = False
        st.session_state.selected_facilities = ["All Facilities"]
        st.session_state.selected_regions = []
        st.session_state.selection_applied = True
        st.session_state.facility_filter_applied = False
        st.session_state.last_computed_kpis = None
        st.session_state.last_computed_newborn_kpis = None
        st.session_state.summary_kpi_cache = {}
        st.session_state.data_initialized = False
        for tab in st.session_state.tab_initialized.keys():
            st.session_state.tab_initialized[tab] = False
        st.session_state.tab_data_loaded["maternal"] = True
        st.session_state.tab_data_loaded["newborn"] = True
        st.session_state.tab_data_loaded["summary"] = False
        st.session_state.tab_data_loaded["mentorship"] = False
        st.session_state.tab_data_loaded["data_quality"] = True
        st.session_state.tab_loading["summary"] = False
        st.session_state.tab_loading["mentorship"] = False
        st.session_state.show_summarized_data = True
    else:
        st.session_state.user_changed = False


# Initialize session state at the very beginning
initialize_session_state()


def count_unique_patients(patient_df, facility_uids, org_unit_column="orgUnit"):
    """Count unique patients from patient-level dataframe using UIDs"""
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
    """Calculate maternal indicators from MATERNAL patient-level data ONLY"""
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

    # Count total maternal patients (each patient is a delivery)
    total_deliveries = filtered_df["tei_id"].nunique()

    # IMPORTANT: Use KPI utilities for proper calculation
    # These functions handle the specific columns for maternal data
    from utils.kpi_utils import (
        compute_maternal_death_count,
        compute_stillbirth_count,
        compute_birth_counts,
        compute_total_deliveries,
        compute_kpis,
    )

    # Compute using KPI utilities
    kpi_data = compute_kpis(filtered_df, facility_uids)

    # Extract values from KPI data
    maternal_deaths = kpi_data.get("maternal_deaths", 0)
    stillbirths = kpi_data.get("stillbirths", 0)
    live_births = kpi_data.get("live_births", 0)
    total_births = kpi_data.get("total_births", 0)

    # Calculate rates
    maternal_death_rate = (
        (maternal_deaths / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )
    stillbirth_rate = (
        (stillbirths / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )

    return {
        "total_deliveries": total_deliveries,
        "maternal_deaths": int(maternal_deaths),
        "maternal_death_rate": round(maternal_death_rate, 2),
        "live_births": int(live_births),
        "stillbirths": int(stillbirths),
        "total_births": int(total_births),
        "stillbirth_rate": round(stillbirth_rate, 2),
        "low_birth_weight_rate": 0.0,  # Will be calculated from newborn data
        "low_birth_weight_count": 0,
    }


def calculate_newborn_indicators_from_patients(patient_df, facility_uids):
    """Calculate newborn indicators from NEWBORN patient-level data ONLY"""
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

    # Calculate NMR using newborn-specific function
    nmr_data = compute_nmr_kpi(filtered_df, facility_uids)

    # Calculate KMC coverage
    kmc_cases = 0
    if "interventions_kmc_administered" in filtered_df.columns:
        kmc_cases = filtered_df[
            filtered_df["interventions_kmc_administered"] == 1
        ].shape[0]
    kmc_coverage_rate = (kmc_cases / total_admitted * 100) if total_admitted > 0 else 0

    # Count low birth weight newborns - FROM NEWBORN DATA
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
        "nmr": nmr_data.get("nmr_per_1000", 0.0),
        "nmr_raw": nmr_data.get("nmr_per_1000", 0.0),
        "nmr_dead_count": nmr_data.get("dead_count", 0),
        "nmr_total_admitted": total_admitted,
        "kmc_coverage_rate": round(kmc_coverage_rate, 2),
        "kmc_cases": kmc_cases,
        "total_lbw": total_lbw,
    }


def get_location_display_name(
    filter_mode, selected_regions, selected_facilities, country_name
):
    """Optimized location display name generation"""
    if filter_mode == "All Facilities":
        return country_name, "Country"
    elif filter_mode == "By Region" and selected_regions:
        return (
            (
                selected_regions[0]
                if len(selected_regions) == 1
                else ", ".join(selected_regions)
            ),
            "Region" if len(selected_regions) == 1 else "Regions",
        )
    elif filter_mode == "By Facility" and selected_facilities:
        return (
            (
                selected_facilities[0]
                if len(selected_facilities) == 1
                else ", ".join(selected_facilities)
            ),
            "Facility" if len(selected_facilities) == 1 else "Facilities",
        )
    else:
        return country_name, "Country"


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


def update_facility_selection(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
):
    """Optimized facility selection update"""
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif filter_mode == "By Region" and selected_regions:
        facility_uids = []
        display_names = []
        for region in selected_regions:
            if region in facilities_by_region:
                facility_uids.extend(
                    fac_uid for _, fac_uid in facilities_by_region[region]
                )
                display_names.append(region)
        comparison_mode = "region"
    elif filter_mode == "By Facility" and selected_facilities:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    else:
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    return facility_uids, display_names, comparison_mode


def render_summary_dashboard_shared(
    user, country_name, facilities_by_region, facility_mapping, shared_data
):
    """Optimized Summary Dashboard using patient-level data - FIXED INDICATOR SOURCES"""

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
                Get comprehensive overview of maternal and newborn health indicators across all facilities
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
                key="load_summary_data",
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

    logging.info("National summary dashboard rendering - USING PATIENT-LEVEL DATA")

    # Get facility selection
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # Update facility selection
    facility_uids, display_names, comparison_mode = update_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Get location display name
    location_name, location_type = get_location_display_name(
        filter_mode, selected_regions, selected_facilities, country_name
    )

    # Use patient-level data
    maternal_data = shared_data["maternal"]
    newborn_data = shared_data["newborn"]

    if not maternal_data and not newborn_data:
        st.error("No data available for summary dashboard")
        return

    # Get patient dataframes - IMPORTANT: SEPARATE DATA SOURCES
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
    else:
        # Compute summary data using patient-level data - FIXED: SEPARATE SOURCES
        with st.spinner("Computing summary statistics from patient-level data..."):
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

            # FIX: Calculate maternal indicators from MATERNAL DATA ONLY
            maternal_indicators = calculate_maternal_indicators_from_patients(
                maternal_patients, facility_uids
            )

            # FIX: Calculate newborn indicators from NEWBORN DATA ONLY
            newborn_indicators = calculate_newborn_indicators_from_patients(
                newborn_patients, facility_uids
            )

            # Regional comparison using patient data
            regional_comparison_data = {}
            for region_name, facilities in facilities_by_region.items():
                region_facility_uids = [fac_uid for fac_name, fac_uid in facilities]

                # Count maternal patients from MATERNAL DATA
                if (
                    not maternal_patients.empty
                    and "orgUnit" in maternal_patients.columns
                ):
                    region_maternal_patients = maternal_patients[
                        maternal_patients["orgUnit"].isin(region_facility_uids)
                    ]
                    maternal_count = region_maternal_patients["tei_id"].nunique()
                else:
                    maternal_count = 0

                # Count newborn patients from NEWBORN DATA
                if not newborn_patients.empty and "orgUnit" in newborn_patients.columns:
                    region_newborn_patients = newborn_patients[
                        newborn_patients["orgUnit"].isin(region_facility_uids)
                    ]
                    newborn_count = region_newborn_patients["tei_id"].nunique()
                else:
                    newborn_count = 0

                regional_comparison_data[region_name] = {
                    "mothers": maternal_count,
                    "newborns": newborn_count,
                }

            summary_data = {
                "maternal_indicators": maternal_indicators,
                "newborn_indicators": newborn_indicators,
                "maternal_tei_count": maternal_patient_count,
                "newborn_tei_count": newborn_patient_count,
                "newborn_start_date": newborn_start_date,
                "maternal_start_date": maternal_start_date,
                "regional_comparison_data": regional_comparison_data,
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
    regional_comparison_data = summary_data["regional_comparison_data"]

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

    # DIRECTLY SHOW TABLES WITHOUT SUMMARY BUTTON
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

    # Newborn Overview Table - USING NEWBORN DATA ONLY
    st.markdown("### 👶 Newborn Care Overview")

    # Create newborn table data with actual NMR value from NEWBORN DATA
    newborn_table_data = {
        "No": list(range(1, 8)),
        "Indicator": [
            "Start Date",
            location_type,
            "Total Admitted Newborns",
            "NMR (per 1000)",
            "KMC Coverage (%)",
            "Low Birth Weight Count (<2500g)",
            "Low Birth Weight Rate (%)",
        ],
        "Value": [
            newborn_start_date,
            location_name,
            f"{newborn_tei_count:,}",
            f"{newborn_indicators['nmr']:.2f}",
            f"{newborn_indicators['kmc_coverage_rate']:.2f}%",
            f"{newborn_indicators['total_lbw']:,}",
            f"{(newborn_indicators['total_lbw'] / newborn_tei_count * 100) if newborn_tei_count > 0 else 0:.2f}%",
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
            "KMC Coverage (%)": [f"{newborn_indicators['kmc_coverage_rate']:.2f}"],
            "Low Birth Weight Count": [newborn_indicators["total_lbw"]],
            "Low Birth Weight Rate (%)": [
                f"{(newborn_indicators['total_lbw'] / newborn_tei_count * 100) if newborn_tei_count > 0 else 0:.2f}"
            ],
        }
        newborn_df = pd.DataFrame(newborn_data_download)
        st.download_button(
            "Download Newborn Data",
            data=newborn_df.to_csv(index=False),
            file_name=f"newborn_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Maternal Overview Table - USING MATERNAL DATA ONLY
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
            "Maternal Death Rate (%)",
            "Live Births",
            "Stillbirths",
            "Stillbirth Rate (%)",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_indicators['total_deliveries']:,}",
            f"{maternal_indicators['maternal_death_rate']:.2f}%",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['stillbirth_rate']:.2f}%",
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
            "Maternal Death Rate (%)": [
                f"{maternal_indicators['maternal_death_rate']:.2f}"
            ],
            "Live Births": [maternal_indicators["live_births"]],
            "Stillbirths": [maternal_indicators["stillbirths"]],
            "Stillbirth Rate (%)": [f"{maternal_indicators['stillbirth_rate']:.2f}"],
        }
        maternal_df = pd.DataFrame(maternal_data_download)
        st.download_button(
            "Download Maternal Data",
            data=maternal_df.to_csv(index=False),
            file_name=f"maternal_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Regional comparison table
    st.markdown("---")
    st.markdown("### 📊 Mothers & Newborns by region")

    if regional_comparison_data:
        regions = list(regional_comparison_data.keys())
        total_mothers = sum(
            data["mothers"] for data in regional_comparison_data.values()
        )
        total_newborns = sum(
            data["newborns"] for data in regional_comparison_data.values()
        )

        # Create transposed table structure
        transposed_data = []
        for i, region in enumerate(regions, 1):
            transposed_data.append(
                {
                    "No": i,
                    "Region Name": region,
                    "Admitted Mothers": f"{regional_comparison_data[region]['mothers']:,}",
                    "Admitted Newborns": f"{regional_comparison_data[region]['newborns']:,}",
                }
            )

        # Add TOTAL row
        transposed_data.append(
            {
                "No": "",
                "Region Name": "TOTAL",
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

        # Add download button for regional data
        col_info3, col_download3 = st.columns([3, 1])
        with col_download3:
            download_data = []
            for region, data in regional_comparison_data.items():
                download_data.append(
                    {
                        "Region": region,
                        "Admitted Mothers": data["mothers"],
                        "Admitted Newborns": data["newborns"],
                    }
                )
            download_data.append(
                {
                    "Region": "TOTAL",
                    "Admitted Mothers": total_mothers,
                    "Admitted Newborns": total_newborns,
                }
            )
            download_df = pd.DataFrame(download_data)
            st.download_button(
                "Download Regional Data",
                data=download_df.to_csv(index=False),
                file_name=f"regional_comparison_{country_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No regional data available for comparison.")


def render_maternal_dashboard_shared(
    user,
    maternal_data,
    country_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",
):
    """Optimized Maternal Dashboard rendering using patient-level data with UID filtering"""

    # Only run if this is the active tab
    if st.session_state.active_tab != "maternal":
        return

    logging.info("Maternal dashboard rendering with patient-level data")

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

    # Update facility selection
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    facility_uids, display_names, comparison_mode = update_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Use patient data directly
    working_df = patients_df.copy()

    # Filter by UID EARLY
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # Ensure ALL patients have a date for filtering
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
        else:
            working_df["event_date"] = pd.NaT

    # Ensure enrollment_date exists and is datetime
    if "enrollment_date" in working_df.columns:
        working_df["enrollment_date"] = pd.to_datetime(
            working_df["enrollment_date"], errors="coerce"
        )
    else:
        working_df["enrollment_date"] = working_df["event_date"]

    # Create a combined date that always has a value
    working_df["combined_date"] = working_df["event_date"].combine_first(
        working_df["enrollment_date"]
    )

    # Store the original df for KPI calculations
    st.session_state.maternal_patients_df = working_df.copy()

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # Optimized header rendering
    if filter_mode == "All Facilities":
        header_title = f"🤰 Maternal Inpatient Data - {country_name}"
        header_subtitle = f"all {len(facility_mapping)} facilities"
    elif filter_mode == "By Region" and display_names:
        if len(display_names) == 1:
            header_title = f"🤰 Maternal Inpatient Data - {display_names[0]} Region"
        else:
            header_title = "🤰 Maternal Inpatient Data - Multiple Regions"
        header_subtitle = f"{len(facility_uids)} facilities"
    elif filter_mode == "By Facility" and display_names:
        if len(display_names) == 1:
            header_title = f"🤰 Maternal Inpatient Data - {display_names[0]}"
        else:
            header_title = "🤰 Maternal Inpatient Data - Multiple Facilities"
        header_subtitle = f"{len(display_names)} facilities"
    else:
        header_title = f"🤰 Maternal Inpatient Data - {country_name}"
        header_subtitle = f"all {len(facility_mapping)} facilities"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Displaying data from {header_subtitle}**")

    # Progress container
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("---")
        st.markdown("### 📈 Preparing Dashboard...")
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.markdown(
                """
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">Processing Data</h4>
            <p style="margin: 5px 0; font-size: 14px;">• Computing KPIs and indicators...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Generating charts and visualizations...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Preparing data tables...</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">This may take 2-4 minutes depending on data size</p>
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
        filters = render_patient_filter_controls(
            working_df, container=col_ctrl, context="national_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply date filters FIRST to get the correct time period
    filtered_for_all = apply_patient_filters(working_df, filters, facility_uids)

    # Store BOTH versions
    st.session_state["filtered_patients"] = filtered_for_all.copy()
    st.session_state["all_patients_for_kpi"] = filtered_for_all.copy()

    # KPI Cards with FILTERED data
    with kpi_container:
        location_name, location_type = get_location_display_name(
            filter_mode, selected_regions, selected_facilities, country_name
        )

        user_id = str(user.get("id", user.get("username", "default_user")))
        kpi_data = render_kpi_cards(filtered_for_all, location_name, user_id=user_id)

        # Save for summary dashboard to reuse
        st.session_state.last_computed_kpis = kpi_data
        st.session_state.last_computed_facilities = facility_uids
        st.session_state.last_computed_timestamp = time.time()

    # CLEAR THE PROGRESS INDICATOR ONCE KPI CARDS ARE DONE
    progress_container.empty()

    # Charts section
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        selected_kpi = render_kpi_tab_navigation()

        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - {comparison_mode.title()} Comparison - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            render_comparison_chart(
                kpi_selection=selected_kpi,
                patient_df=filtered_for_all,
                comparison_mode=comparison_mode,
                display_names=display_names,
                facility_uids=facility_uids,
                facilities_by_region=facilities_by_region,
                region_names=display_names if comparison_mode == "region" else None,
                bg_color=bg_color,
                text_color=text_color,
                is_national=True,
                filtered_patients=filtered_for_all,
            )
        else:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
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
    """Main optimized render function"""
    st.set_page_config(
        page_title="National Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Re-initialize session state for safety
    initialize_session_state()

    # Show user change notification if needed
    if st.session_state.get("user_changed", False):
        st.sidebar.info("User changed - loading fresh data...")
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

    try:
        with open("utils/national.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Get user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    country_name = user.get("country_name", "Unknown Country")

    # Compact sidebar user info
    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 {username}</div>
            <div>🌍 {country_name}</div>
            <div>🛡️ {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh Data Button
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        clear_shared_cache(user)
        st.session_state.refresh_trigger = True
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
            with st.spinner("Loading facility data..."):
                static_data = get_static_data(user)
                st.session_state.facilities_by_region = static_data[
                    "facilities_by_region"
                ]
                st.session_state.facility_mapping = static_data["facility_mapping"]
                st.session_state.program_uid_map = static_data["program_uid_map"]
                st.session_state.static_data_loaded = True
                st.session_state.user_changed = False

    facilities_by_region = st.session_state.facilities_by_region
    facility_mapping = st.session_state.facility_mapping
    program_uid_map = st.session_state.program_uid_map

    # SINGLE DATA LOADING - Only load once and store in variable
    if not st.session_state.get("data_initialized", False):
        # First time or fresh data needed
        with st.spinner("Loading dashboard data..."):
            shared_data = get_shared_program_data_optimized(
                user, program_uid_map, show_spinner=False
            )
            st.session_state.data_initialized = True
            st.session_state.cached_shared_data = shared_data
            logging.info("Initial data loading complete")
    else:
        # Use cached data from session state - no loading needed
        shared_data = st.session_state.cached_shared_data
        logging.info("Using cached shared data from session state")

    # Add cache status indicator in sidebar
    if st.session_state.get("data_initialized", False):
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        timestamp_key = f"shared_data_timestamp_{user_key}"

        if timestamp_key in st.session_state:
            time_elapsed = time.time() - st.session_state[timestamp_key]
            minutes_old = int(time_elapsed // 60)
            seconds_old = int(time_elapsed % 60)

            if minutes_old < 30:
                st.sidebar.info(f"Data: {minutes_old}m {seconds_old}s old")
            else:
                st.sidebar.warning(f"Data: {minutes_old}m old (will auto-refresh)")

    # ================ COMPACT FACILITY SELECTION ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">🏥 Facility Selection</p>',
        unsafe_allow_html=True,
    )

    # Filter mode selection
    radio_options = ["All Facilities", "By Region", "By Facility"]
    try:
        current_index = radio_options.index(st.session_state.filter_mode)
    except ValueError:
        current_index = 0
        st.session_state.filter_mode = "All Facilities"

    new_filter_mode = st.sidebar.radio(
        "Select by:",
        radio_options,
        index=current_index,
        key="mode_radio",
        label_visibility="collapsed",
    )

    if new_filter_mode != st.session_state.filter_mode:
        st.session_state.filter_mode = new_filter_mode
        st.session_state.selected_regions = []
        st.session_state.selected_facilities = []
        st.session_state.facility_filter_applied = False

    # Ultra-compact selection form
    with st.sidebar.form("selection_form", border=False):
        temp_selected_regions = st.session_state.selected_regions.copy()
        temp_selected_facilities = st.session_state.selected_facilities.copy()

        if st.session_state.filter_mode == "By Region":
            region_options = {
                f"{region} ({len(facilities_by_region.get(region, []))} fac)": region
                for region in facilities_by_region.keys()
            }
            selected_region_labels = st.multiselect(
                "Choose regions:",
                options=list(region_options.keys()),
                default=[
                    label
                    for label in region_options.keys()
                    if region_options[label] in st.session_state.selected_regions
                ],
                help="Select regions",
                key="region_multiselect",
                label_visibility="collapsed",
            )
            temp_selected_regions = [
                region_options[label] for label in selected_region_labels
            ]

        elif st.session_state.filter_mode == "By Facility":
            # Simple facility selection
            all_facilities = []
            for region_name, facilities in facilities_by_region.items():
                all_facilities.extend([fac_name for fac_name, _ in facilities])

            facility_options = ["All Facilities"] + all_facilities
            selected_facilities = st.multiselect(
                "Choose facilities:",
                options=facility_options,
                default=st.session_state.selected_facilities,
                help="Select facilities",
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
                    selected_facilities = all_facilities.copy()

            temp_selected_facilities = selected_facilities

        selection_submitted = st.form_submit_button(
            "Apply Selection", use_container_width=True
        )
        if selection_submitted:
            st.session_state.selected_regions = temp_selected_regions
            st.session_state.selected_facilities = temp_selected_facilities
            st.session_state.selection_applied = True
            st.session_state.facility_filter_applied = True
            st.rerun()

    # Display selection summary
    total_facilities = len(facility_mapping)
    if st.session_state.filter_mode == "All Facilities":
        display_text = f"Selected: All ({total_facilities})"
    elif (
        st.session_state.filter_mode == "By Region"
        and st.session_state.selected_regions
    ):
        display_text = f"Selected: {len(st.session_state.selected_regions)} regions"
    elif (
        st.session_state.filter_mode == "By Facility"
        and st.session_state.selected_facilities
    ):
        display_text = f"Selected: {len(st.session_state.selected_facilities)} / {total_facilities} facilities"
    else:
        display_text = f"Selected: All ({total_facilities})"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: -10px;'>{display_text}</p>",
        unsafe_allow_html=True,
    )

    # ================ COMPACT VIEW MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">📊 View Mode</p>',
        unsafe_allow_html=True,
    )

    view_mode = "Normal Trend"
    if st.session_state.filter_mode != "All Facilities":
        view_mode = st.sidebar.radio(
            "View:",
            ["Normal Trend", "Comparison View"],
            index=0,
            key="view_mode_national",
            label_visibility="collapsed",
        )

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
            logging.info("Switched to Maternal tab")

        maternal_data = shared_data["maternal"]
        if maternal_data:
            render_maternal_dashboard_shared(
                user,
                maternal_data,
                country_name,
                facilities_by_region,
                facility_mapping,
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
            # Need to update render_newborn_dashboard to accept patient-level data
            render_newborn_dashboard(
                user=user,
                program_uid=program_uid_map.get("Newborn Care Form"),
                country_name=country_name,
                facilities_by_region=facilities_by_region,
                facility_mapping=facility_mapping,
                view_mode=view_mode,
                shared_newborn_data=newborn_data,
            )
        else:
            st.error("Newborn data not available")

    with tab3:
        if st.session_state.active_tab != "summary":
            st.session_state.active_tab = "summary"
            logging.info("Switched to Summary tab")

        render_summary_dashboard_shared(
            user,
            country_name,
            facilities_by_region,
            facility_mapping,
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
                    key="load_mentorship_data",
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
        if st.session_state.active_tab != "data_quality":
            st.session_state.active_tab = "data_quality"
            logging.info("Switched to Data Quality tab")

        render_data_quality_tracking(user)

    # Log current active tab state
    logging.info(f"Current active tab: {st.session_state.active_tab}")

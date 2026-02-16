# dashboards/regional.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from datetime import datetime
from utils.resource import render_resources_tab
from newborns_dashboard.reginal_newborn import (
    render_newborn_dashboard_shared,
)
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
)
from utils.kpi_utils import clear_cache, compute_kpis
from utils.odk_dashboard import display_odk_dashboard
from components.edit_profile import render_edit_profile
from utils.usage_tracking import render_usage_tracking_shared
#from dashboards.data_quality_tracking import render_data_quality_tracking


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
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
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
            "tracking": False,
        },
        "tab_data_loaded": {
            "maternal": True,
            "newborn": True,
            "summary": False,
            "mentorship": False,
            "data_quality": True,
            "tracking": True,
        },
        "tab_loading": {
            "summary": False,
            "mentorship": False,
        },
        "show_summarized_data": True,
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
        st.session_state.static_data_loaded = False
        st.session_state.selected_facilities = ["All Facilities"]
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


def get_location_display_name(selected_facilities, region_name):
    """Optimized location display name generation - SIMPLIFIED FOR REGIONAL"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        return f"{len(selected_facilities)} Facilities", "Facilities"


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


def update_facility_selection(selected_facilities, facility_mapping):
    """Simplified facility selection update - REGIONAL ONLY"""
    if selected_facilities == ["All Facilities"]:
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif selected_facilities:
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


def render_data_quality_maintenance_message():
    """Display maintenance message for data quality dashboard"""
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #fff8e1, #ffecb3);
         border-radius: 12px; border: 2px solid #ffb300; margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔧</div>
        <h2 style="color: #e65100; margin-bottom: 1rem;">Data Quality Dashboard Under Maintenance</h2>
        <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            The data quality dashboard is currently undergoing maintenance and will be available soon.
        </p>
        <div style="display: inline-block; padding: 10px 20px; background: #ff9800; color: white;
             border-radius: 25px; font-weight: bold;">
            Maintenance in Progress
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_summary_dashboard_shared(
    user, region_name, facility_mapping, selected_facilities, shared_data
):
    """Optimized Summary Dashboard - SIMPLIFIED with only 3 key indicators + FACILITY COMPARISON TABLE"""

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
                Get overview of key maternal and newborn health indicators across selected facilities
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

    logging.info(
        "Regional summary dashboard rendering - WITH FACILITY COMPARISON TABLE"
    )

    # Get location display name
    location_name, location_type = get_location_display_name(
        selected_facilities, region_name
    )

    # Get facility UIDs
    facility_uids, display_names, comparison_mode = update_facility_selection(
        selected_facilities, facility_mapping
    )

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
        cache_key in st.session_state.summary_kpi_cache
        and time.time() - st.session_state.summary_kpi_cache[cache_key]["timestamp"]
        < 300
    ):
        summary_data = st.session_state.summary_kpi_cache[cache_key]["data"]
    else:
        # Compute summary data using patient-level data
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
                    # compute_hypothermia_on_admission_rate,  # For Hypothermia on Admission
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
                # hypothermia_rate, hypo_count, total_hypo = (
                #     compute_hypothermia_on_admission_rate(
                #         filtered_newborn, facility_uids
                #     )
                # )

                newborn_kpis = {
                    "admitted_newborns_count": admitted_newborns_count,
                    "neonatal_mortality_rate": nmr_rate,
                    # "hypothermia_rate": hypothermia_rate,
                    "death_count": death_count,
                    "total_deaths": total_deaths,
                    # "hypothermia_count": hypo_count,
                    # "total_hypo": total_hypo,
                }

            # =========== NEW: FACILITY COMPARISON DATA ===========
            # For regional level, show facility-level data (like national shows regional)
            facility_comparison_data = {}

            # Determine which facilities to include
            if selected_facilities == ["All Facilities"]:
                # Include all facilities in the region
                facilities_to_check = list(facility_mapping.keys())
            else:
                # Only include selected facilities
                facilities_to_check = selected_facilities

            for facility_name in facilities_to_check:
                if facility_name in facility_mapping:
                    facility_uid = facility_mapping[facility_name]

                    # Maternal count for facility (Admitted Mothers)
                    maternal_count = 0
                    if (
                        not maternal_patients.empty
                        and "orgUnit" in maternal_patients.columns
                    ):
                        facility_maternal_data = maternal_patients[
                            maternal_patients["orgUnit"] == facility_uid
                        ].copy()

                        # Use the same logic as national level to get Admitted Mothers count
                        from utils.kpi_admitted_mothers import (
                            get_numerator_denominator_for_admitted_mothers,
                        )

                        # Get date range filters if available
                        date_range_filters = {}
                        if "filters" in st.session_state:
                            date_range_filters = {
                                "start_date": st.session_state.filters.get(
                                    "start_date"
                                ),
                                "end_date": st.session_state.filters.get("end_date"),
                            }

                        numerator, denominator, _ = (
                            get_numerator_denominator_for_admitted_mothers(
                                facility_maternal_data,
                                [facility_uid],
                                date_range_filters,
                            )
                        )
                        maternal_count = numerator

                    # Newborn count for facility - ADMITTED NEWBORNS (from enrollment date)
                    newborn_count = 0
                    if (
                        not newborn_patients.empty
                        and "orgUnit" in newborn_patients.columns
                    ):
                        facility_newborn_data = newborn_patients[
                            newborn_patients["orgUnit"] == facility_uid
                        ].copy()

                        # Count Admitted Newborns using enrollment date
                        from newborns_dashboard.kpi_utils_newborn import (
                            compute_admitted_newborns_count,
                        )

                        newborn_count = compute_admitted_newborns_count(
                            facility_newborn_data, [facility_uid]
                        )

                    # Only include facilities with data
                    if maternal_count > 0 or newborn_count > 0:
                        facility_comparison_data[facility_name] = {
                            "mothers": maternal_count,  # Admitted Mothers
                            "newborns": newborn_count,  # Admitted Newborns (from enrollment)
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
                # "hypothermia_rate": newborn_kpis.get(
                #     "hypothermia_rate", 0.0
                # ),  # 3. Hypothermia After Admission
                "newborn_start_date": newborn_start_date,
                # NEW: Facility comparison data
                "facility_comparison_data": facility_comparison_data,
                "location_name": location_name,
                "location_type": location_type,
            }

            # Cache the computed data
            st.session_state.summary_kpi_cache[cache_key] = {
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
    # hypothermia_rate = summary_data["hypothermia_rate"]
    newborn_start_date = summary_data["newborn_start_date"]

    facility_comparison_data = summary_data["facility_comparison_data"]
    location_name = summary_data["location_name"]
    location_type = summary_data["location_type"]

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
    .maternal-table thead tr { background: linear-gradient(135deg, #2ca02c, #228b22) !important; }
    .newborn-table thead tr { background: linear-gradient(135deg, #ff7f0e, #e66a00) !important; }
    .summary-table td:first-child { font-weight: 600; color: #666; text-align: center; width: 40px; }
    .summary-table th:first-child { text-align: center; width: 40px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

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
        # (
        #     col3,
        #     "Hypothermia on Admission",
        #     f"{hypothermia_rate:.2f}%",
        #     "Newborns with temp < 36.5°C on admission",
        #     "#2ca02c",
        # ),
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

    st.markdown("---")

    # =========== NEW: FACILITY COMPARISON TABLE ===========
    st.markdown("### 📊 Facility Summary")

    if facility_comparison_data:
        facilities = list(facility_comparison_data.keys())
        total_mothers = sum(
            data["mothers"] for data in facility_comparison_data.values()
        )
        total_newborns = sum(
            data["newborns"] for data in facility_comparison_data.values()
        )

        # Sort facilities alphabetically for better organization
        facilities = sorted(facilities)

        # Create table structure with 4 columns
        table_data = []
        for i, facility in enumerate(facilities, 1):
            mother_count = facility_comparison_data[facility]["mothers"]
            newborn_count = facility_comparison_data[facility]["newborns"]

            table_data.append(
                {
                    "No": str(i),
                    "Facility Name": facility,
                    "Total Admitted Mothers": (
                        f"{mother_count:,}"
                        if isinstance(mother_count, int)
                        else str(mother_count)
                    ),
                    "Total Admitted Newborns": (
                        f"{newborn_count:,}"
                        if isinstance(newborn_count, int)
                        else str(newborn_count)
                    ),
                }
            )

        # Add TOTAL row
        table_data.append(
            {
                "No": "-",
                "Facility Name": "TOTAL",
                "Total Admitted Mothers": f"{total_mothers:,}",
                "Total Admitted Newborns": f"{total_newborns:,}",
            }
        )

        # Create DataFrame
        table_df = pd.DataFrame(table_data)

        # Display the table
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Add download button for facility data
        col_info, col_download = st.columns([3, 1])
        with col_download:
            download_data = []
            for facility, data in facility_comparison_data.items():
                download_data.append(
                    {
                        "Facility": facility,
                        "Total Admitted Mothers": (
                            data["mothers"] if isinstance(data["mothers"], int) else 0
                        ),
                        "Total Admitted Newborns": (
                            data["newborns"] if isinstance(data["newborns"], int) else 0
                        ),
                    }
                )
            download_data.append(
                {
                    "Facility": "TOTAL",
                    "Total Admitted Mothers": total_mothers,
                    "Total Admitted Newborns": total_newborns,
                }
            )
            download_df = pd.DataFrame(download_data)
            st.download_button(
                "📥 Download Facility Data",
                data=download_df.to_csv(index=False),
                file_name=f"facility_summary_{region_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No facility data available for comparison.")


def render_maternal_dashboard_shared(
    user,
    maternal_data,
    region_name,
    selected_facilities,
    facility_uids,
    view_mode="Normal Trend",
    facility_mapping=None,
    facility_names=None,
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

    # Use patient data directly
    working_df = patients_df.copy()

    # Filter by UID EARLY
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # =========== CRITICAL: USE normalize_patient_dates ===========
    # Use the same function as national.py to ensure consistency
    from utils.dash_co import normalize_patient_dates

    # This will create proper enrollment_date column
    working_df = normalize_patient_dates(working_df)

    if working_df.empty:
        st.info("ℹ️ No data available for the selected facilities/indicators.")
        return

    # Log date statistics
    valid_dates = working_df["enrollment_date"].notna().sum()
    total_patients = len(working_df)
    logging.info(
        f"📅 REGIONAL: enrollment_date - {valid_dates}/{total_patients} valid dates"
    )

    # Log sample dates to verify they're correct
    if valid_dates > 0:
        sample_dates = working_df["enrollment_date"].dropna().head(3).tolist()
        logging.info(f"📅 REGIONAL: Sample dates: {sample_dates}")
    # =========== END OF CRITICAL ADDITION ===========

    # Get current KPI selection to know which date column we need
    current_kpi = st.session_state.get(
        "selected_kpi", "Maternal Death Rate (per 100,000)"
    )
    from utils.kpi_utils import get_relevant_date_column_for_kpi

    kpi_date_column = get_relevant_date_column_for_kpi(current_kpi)

    # Log which date column will be used
    if kpi_date_column and kpi_date_column in working_df.columns:
        valid_dates = working_df[kpi_date_column].notna().sum()
        total_patients = len(working_df)
        logging.info(
            f"📅 REGIONAL: KPI '{current_kpi}' will use date column: {kpi_date_column}"
        )
        logging.info(
            f"📅 REGIONAL: Date stats for {kpi_date_column}: {valid_dates}/{total_patients} valid dates"
        )
    else:
        logging.warning(
            f"⚠️ REGIONAL: KPI date column '{kpi_date_column}' not found in data"
        )

    # Store the original df for KPI calculations
    st.session_state.maternal_patients_df = working_df.copy()

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
        f'<div class="main-header" style="margin-bottom: 0.1rem; line-height: 1.2;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"<div style='margin-bottom: 0.1rem;'><b>Displaying data from {header_subtitle}</b></div>", unsafe_allow_html=True)

    # Create containers for better performance
    # kpi_container = st.container() # Removed empty container

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_patient_filter_controls(
            working_df, container=col_ctrl, context="regional_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply date filters FIRST to get the correct time period
    logging.info(
        f"🔍 REGIONAL: Calling apply_patient_filters with {len(working_df)} patients"
    )
    logging.info(f"   - Quick range: {filters.get('quick_range', 'N/A')}")
    if "start_date" in filters and "end_date" in filters:
        logging.info(
            f"   - Date range: {filters['start_date']} to {filters['end_date']}"
        )
    logging.info(f"   - Period label: {filters.get('period_label', 'Monthly')}")
    logging.info(f"   - Facility UIDs: {len(facility_uids)}")

    filtered_for_all = apply_patient_filters(working_df, filters, facility_uids)

    logging.info(
        f"🔍 REGIONAL: After apply_patient_filters: {len(filtered_for_all)} patients"
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

        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            render_comparison_chart(
                kpi_selection=selected_kpi,
                patient_df=filtered_for_all,
                comparison_mode="facility",
                display_names=facility_names or selected_facilities,
                facility_uids=facility_uids,
                facilities_by_region=None,
                region_names=None,
                bg_color=bg_color,
                text_color=text_color,
                is_national=False,
                filtered_patients=filtered_for_all,
            )
        else:
            render_trend_chart_section(
                selected_kpi,
                filtered_for_all,
                facility_uids,
                facility_names or selected_facilities,
                bg_color,
                text_color,
                comparison_mode="facility",
                facilities_by_region=None,
                region_names=None,
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
    # st.set_page_config removed - only app.py should call this

    # Re-initialize session state for safety
    initialize_session_state()

    if "edit_profile_mode" not in st.session_state:
        st.session_state.edit_profile_mode = False

    user = st.session_state.get("user", {})
    if st.session_state.edit_profile_mode:
        render_edit_profile(user, view_state_key="edit_profile_mode", key_prefix="regional")
        return

    # Show user change notification if needed
    if st.session_state.get("user_changed", False):
        st.sidebar.info("User changed - loading fresh data...")
        current_user = st.session_state.get("user", {})
        clear_shared_cache(current_user)

    # Load optimized CSS
    st.markdown(
        """
    <style>
    /* Premium Professional Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Card-based layout for better depth */
    .form-container, [data-testid="stExpander"], .stDataFrame, .stPlotlyChart {
        background-color: white !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 1rem !important;
    }

    .main-header { font-size: 1.5rem !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; color: #0f172a !important; }
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
    .filter-box { background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 5px; }
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

    if st.sidebar.button("Edit Profile", use_container_width=True):
        st.session_state.edit_profile_mode = True
        st.rerun()

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
                st.session_state.facilities = static_data["facilities"]
                st.session_state.facility_mapping = static_data["facility_mapping"]
                st.session_state.program_uid_map = static_data["program_uid_map"]
                st.session_state.static_data_loaded = True
                st.session_state.user_changed = False

    facilities = st.session_state.facilities
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
            st.session_state.cached_shared_data_regional = shared_data
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
            # Sidebar info removed per user request

    # ================ SIMPLE FACILITY SELECTION ================
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
            "Apply Selection", use_container_width=True
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

    # ================ SIMPLE VIEW MODE ================
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

    # ================ STRICT TAB BRANCHING (HARD LAZY LOADING) ================
    tab_options = {
        "Maternal": "maternal",
        "Newborn": "newborn",
        "Summary": "summary",
        "Mentorship": "mentorship",
        "Resources": "resources",
        "Usage Tracking": "tracking",
    }
    reverse_tab_options = {v: k for k, v in tab_options.items()}
    current_tab_label = reverse_tab_options.get(
        st.session_state.active_tab, "Maternal"
    )
    selector_key = "regional_active_tab_selector"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = current_tab_label

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
            min-height: auto !important;
            justify-content: flex-start !important;
        }
        .dashboard-tab-header {
            margin: -2.25rem 0 0 0;
            padding: 0.75rem 1rem;
            border: 1px solid #c7d2fe;
            border-radius: 12px;
            background: linear-gradient(135deg, #eff6ff, #e0e7ff);
            color: #1e3a8a;
            font-size: 1.08rem;
            font-weight: 800;
        }
        .st-key-regional_active_tab_selector [data-testid="stRadio"] {
            padding: 0.55rem 0.7rem;
            margin: 0 !important;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            background: #f8fafc;
        }
        .st-key-regional_active_tab_selector [data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {
            gap: 0.45rem;
            margin-top: 0 !important;
        }
        .st-key-regional_active_tab_selector [data-testid="stRadio"] label {
            margin: 0 !important;
            padding: 0.55rem 1rem;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background: #ffffff;
            font-size: 1.03rem !important;
            font-weight: 800 !important;
        }
        .st-key-regional_active_tab_selector [data-testid="stRadio"] label:has(input:checked) {
            border-color: #1d4ed8;
            color: #ffffff !important;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.25);
        }
        /* Newborn tab (2nd option) custom peach theme - robust selectors */
        .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2),
        .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2) label {
            background: #fff7ed !important;
            border-color: #fed7aa !important;
            color: #9a3412 !important;
        }
        .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2):has(input:checked),
        .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2) label:has(input:checked) {
            background: linear-gradient(135deg, #fed7aa, #fdba74) !important;
            border-color: #fb923c !important;
            color: #7c2d12 !important;
            box-shadow: 0 2px 8px rgba(249, 115, 22, 0.28) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="dashboard-tab-header">Dashboard Sections</div>', unsafe_allow_html=True)

    selected_tab_label = st.radio(
        "Dashboard Tab",
        options=list(tab_options.keys()),
        key=selector_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_tab = tab_options[selected_tab_label]

    if st.session_state.active_tab != selected_tab:
        st.session_state.active_tab = selected_tab
        logging.info(f"Switched to {selected_tab.capitalize()} tab")

    if selected_tab == "maternal":
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

    elif selected_tab == "newborn":
        newborn_data = shared_data["newborn"]
        if newborn_data:
            # Create empty facilities_by_region for compatibility
            facilities_by_region = {region_name: [(f[0], f[1]) for f in facilities]}

            render_newborn_dashboard_shared(
                user,
                newborn_data,
                region_name,
                facilities_by_region,
                facility_mapping,
                view_mode=view_mode,
            )
        else:
            st.error("Newborn data not available")

    elif selected_tab == "summary":
        render_summary_dashboard_shared(
            user,
            region_name,
            facility_mapping,
            selected_facilities,
            shared_data,
        )

    elif selected_tab == "mentorship":
        display_odk_dashboard(user)

    elif selected_tab == "resources":
        render_resources_tab()

    elif selected_tab == "tracking":
        # Regional user tracks Facility users in their region
        user_region_id = user.get('region_id')
        render_usage_tracking_shared('regional', user_region_id=user_region_id)
    # Log current active tab state
    logging.info(f"Current active tab: {st.session_state.active_tab}")

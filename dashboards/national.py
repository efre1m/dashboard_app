# dashboards/national.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import time
from components.kpi_card import render_kpi_cards
from newborns_dashboard.national_newborn import render_newborn_dashboard
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_grouped_by_region,
    get_facility_mapping_for_user,
)
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    render_trend_chart_section,
    render_comparison_chart,
    render_additional_analytics,
    get_text_color,
    apply_simple_filters,
    render_simple_filter_controls,
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

# Initialize status system
initialize_status_system()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 2700  # 45 minutes


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
    """Smart data loading with 45-minute auto-refresh and user-specific caching"""
    maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
    newborn_program_uid = program_uid_map.get("Newborn Care Form")

    # Create user-specific session state keys
    user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    shared_loaded_key = f"shared_data_loaded_{user_key}"
    shared_maternal_key = f"shared_maternal_data_{user_key}"
    shared_newborn_key = f"shared_newborn_data_{user_key}"
    shared_timestamp_key = f"shared_data_timestamp_{user_key}"

    current_time = time.time()

    # ✅ INCREASED CACHE TIME: 45 minutes = 2700 seconds
    cache_expired = False
    if shared_timestamp_key in st.session_state:
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        cache_expired = time_elapsed > 2700
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

        def load_data():
            # Use ThreadPoolExecutor with timeout handling
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both data fetching tasks
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

                # ✅ INCREASED TIMEOUT: 300 seconds (5 minutes)
                try:
                    maternal_data = None
                    newborn_data = None

                    # Get maternal data with timeout
                    if maternal_future:
                        maternal_data = maternal_future.result(timeout=300)
                        logging.info(
                            f"✅ Maternal data loaded: {len(maternal_data.get('tei', pd.DataFrame())) if maternal_data else 0} TEIs"
                        )

                    # Get newborn data with timeout
                    if newborn_future:
                        newborn_data = newborn_future.result(timeout=300)
                        logging.info(
                            f"✅ Newborn data loaded: {len(newborn_data.get('tei', pd.DataFrame())) if newborn_data else 0} TEIs"
                        )

                    # Store results
                    st.session_state[shared_maternal_key] = maternal_data
                    st.session_state[shared_newborn_key] = newborn_data
                    st.session_state[shared_loaded_key] = True
                    st.session_state[shared_timestamp_key] = current_time

                    # ✅ STORE in main cached_shared_data for easy access
                    st.session_state.cached_shared_data = {
                        "maternal": maternal_data,
                        "newborn": newborn_data,
                    }

                    # Log fresh data stats
                    maternal_tei_count = (
                        len(maternal_data.get("tei", pd.DataFrame()))
                        if maternal_data
                        else 0
                    )
                    newborn_tei_count = (
                        len(newborn_data.get("tei", pd.DataFrame()))
                        if newborn_data
                        else 0
                    )
                    logging.info(
                        f"✅ FRESH DATA COMPLETE: {maternal_tei_count} maternal TEIs, {newborn_tei_count} newborn TEIs"
                    )

                    # Reset refresh trigger and user changed flags
                    st.session_state.refresh_trigger = False
                    st.session_state.user_changed = False

                    return True

                except concurrent.futures.TimeoutError:
                    logging.error("❌ Data loading timeout after 300 seconds")
                    # Store whatever data we managed to get
                    maternal_data = (
                        maternal_data if "maternal_data" in locals() else None
                    )
                    newborn_data = newborn_data if "newborn_data" in locals() else None

                    st.session_state[shared_maternal_key] = maternal_data
                    st.session_state[shared_newborn_key] = newborn_data
                    st.session_state[shared_loaded_key] = (
                        True  # Mark as loaded even if partial
                    )
                    st.session_state[shared_timestamp_key] = current_time

                    # Store partial data
                    st.session_state.cached_shared_data = {
                        "maternal": maternal_data,
                        "newborn": newborn_data,
                    }

                    logging.warning("⚠️ Using partially loaded data due to timeout")
                    return True  # Continue with partial data

                except Exception as e:
                    logging.error(f"❌ Error during data loading: {e}")
                    return False

        # Show spinner only if requested
        if show_spinner:
            with st.spinner("🚀 Loading dashboard data..."):
                success = load_data()
                if not success:
                    st.error("Data loading failed. Please try refreshing.")
        else:
            success = load_data()

    else:
        # ✅ Use cached data
        maternal_tei_count = (
            len(st.session_state[shared_maternal_key].get("tei", pd.DataFrame()))
            if st.session_state[shared_maternal_key]
            else 0
        )
        newborn_tei_count = (
            len(st.session_state[shared_newborn_key].get("tei", pd.DataFrame()))
            if st.session_state[shared_newborn_key]
            else 0
        )
        time_elapsed = current_time - st.session_state[shared_timestamp_key]
        logging.info(
            f"✅ USING CACHED DATA: {maternal_tei_count} maternal TEIs, {newborn_tei_count} newborn TEIs ({time_elapsed:.0f}s old)"
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
        "refresh_trigger": False,
        "selected_facilities": [],
        "selected_regions": [],
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
        "filter_mode": "All Facilities",
        "filtered_events": pd.DataFrame(),
        "selection_applied": True,
        "cached_events_data": None,
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
            "maternal": True,
            "newborn": True,
            "summary": False,
            "mentorship": False,
            "data_quality": True,
        },
        # ✅ NEW: Track loading state for each tab
        "tab_loading": {
            "summary": False,
            "mentorship": False,
        },
        # ✅ NEW: Force show tables in summary dashboard
        "show_summarized_data": True,
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
        st.session_state.selected_regions = []
        st.session_state.selected_facilities = []
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
        st.session_state.show_summarized_data = True
    else:
        st.session_state.user_changed = False


# Initialize session state at the very beginning
initialize_session_state()


# ✅ CONSISTENT COUNTING FUNCTIONS
def count_unique_newborns_consistent(
    tei_df, facility_uids=None, org_unit_column="tei_orgUnit"
):
    """✅ FIX: Consistent counting of unique newborns across entire dashboard"""
    if tei_df.empty:
        return 0

    # Filter TEI dataframe by the selected facility UIDs if provided
    if facility_uids and org_unit_column in tei_df.columns:
        filtered_tei = tei_df[tei_df[org_unit_column].isin(facility_uids)]
    elif facility_uids and "orgUnit" in tei_df.columns:
        filtered_tei = tei_df[tei_df["orgUnit"].isin(facility_uids)]
    else:
        filtered_tei = tei_df

    # Count unique newborns using the same column consistently
    if "tei_id" in filtered_tei.columns:
        return filtered_tei["tei_id"].nunique()
    elif "trackedEntityInstance" in filtered_tei.columns:
        return filtered_tei["trackedEntityInstance"].nunique()
    else:
        return filtered_tei.index.nunique()


def count_unique_mothers_consistent(enrollments_df, facility_uids=None):
    """✅ FIX: Consistent counting of unique mothers across entire dashboard"""
    if enrollments_df.empty:
        return 0

    # Filter enrollments by facility if provided
    if facility_uids and "orgUnit" in enrollments_df.columns:
        filtered_enrollments = enrollments_df[
            enrollments_df["orgUnit"].isin(facility_uids)
        ]
    else:
        filtered_enrollments = enrollments_df

    # Count unique mothers using the same column consistently
    if "tei_id" in filtered_enrollments.columns:
        return filtered_enrollments["tei_id"].nunique()
    elif "trackedEntityInstance" in filtered_enrollments.columns:
        return filtered_enrollments["trackedEntityInstance"].nunique()
    else:
        return filtered_enrollments.index.nunique()


def count_unique_teis_from_events(
    events_df, facility_uids=None, org_unit_column="orgUnit"
):
    """Count unique TEI IDs from events DataFrame - CONSISTENT WITH REGIONAL"""
    if events_df.empty or "tei_id" not in events_df.columns:
        return 0

    # Filter by facility if specified
    if facility_uids and org_unit_column in events_df.columns:
        filtered_events = events_df[events_df[org_unit_column].isin(facility_uids)]
    else:
        filtered_events = events_df

    return filtered_events["tei_id"].nunique()


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


def filter_data_by_facilities(data_dict, facility_uids):
    """Filter all dataframes in a data dictionary by facility UIDs"""
    if not data_dict or not facility_uids:
        return data_dict

    filtered_data = {}
    for key, df in data_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "orgUnit" in df.columns:
                filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()
            elif "tei_orgUnit" in df.columns:
                filtered_df = df[df["tei_orgUnit"].isin(facility_uids)].copy()
            else:
                filtered_df = df.copy()
            filtered_data[key] = filtered_df
        else:
            filtered_data[key] = df

    return filtered_data


def calculate_regional_comparison_data(
    maternal_events_df,
    newborn_events_df,
    facilities_by_region,
    facility_mapping,
    maternal_enrollments_df=None,
    newborn_tei_df=None,
):
    """✅ FIX: Regional comparison data calculation using EVENTS dataframe (same as kpi_utils)"""
    regional_data = {}

    for region_name, facilities in facilities_by_region.items():
        region_facility_uids = [fac_uid for fac_name, fac_uid in facilities]

        # ✅ FIX: Use events dataframe for counting (same as kpi_utils.compute_total_deliveries())
        if not maternal_events_df.empty and "tei_id" in maternal_events_df.columns:
            if region_facility_uids:
                region_maternal_events = maternal_events_df[
                    maternal_events_df["orgUnit"].isin(region_facility_uids)
                ]
                maternal_count = region_maternal_events["tei_id"].nunique()
            else:
                maternal_count = 0
        else:
            maternal_count = 0

        # ✅ FIX: Use events dataframe for newborns too (consistent approach)
        if not newborn_events_df.empty and "tei_id" in newborn_events_df.columns:
            if region_facility_uids:
                region_newborn_events = newborn_events_df[
                    newborn_events_df["orgUnit"].isin(region_facility_uids)
                ]
                newborn_count = region_newborn_events["tei_id"].nunique()
            else:
                newborn_count = 0
        else:
            newborn_count = 0

        regional_data[region_name] = {
            "mothers": maternal_count,
            "newborns": newborn_count,
        }

    return regional_data


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
            st.session_state.tab_loading[tab_key] = True
            st.session_state.tab_data_loaded[tab_key] = True
            st.rerun()


def render_summary_dashboard_shared(
    user, country_name, facilities_by_region, facility_mapping, shared_data
):
    """✅ FIX: OPTIMIZED Summary Dashboard with CONSISTENT counting - NOW SAME AS KPI_UTILS"""
    if st.session_state.active_tab != "summary":
        return

    # ✅ Check if user has clicked "View Data" button for this tab
    if not st.session_state.tab_data_loaded["summary"]:
        render_tab_placeholder(
            "Summary Dashboard",
            "📊",
            "summary",
            "Get comprehensive overview of maternal and newborn health indicators across all facilities",
        )
        return

    # ✅ Show loading indicator if data is being processed
    if st.session_state.tab_loading["summary"]:
        render_loading_indicator("Summary Dashboard", "📊")
        st.session_state.tab_loading["summary"] = False
        st.rerun()

    logging.info("🔄 Summary dashboard rendering - USING KPI_UTILS LOGIC")

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

    # Use shared data
    maternal_data = shared_data["maternal"]
    newborn_data = shared_data["newborn"]

    if not maternal_data and not newborn_data:
        st.error("No data available for summary dashboard")
        return

    # Create cache key for summary data
    cache_key = f"summary_{location_name}_{len(facility_uids)}"

    # Check if we have cached summary data
    if (
        cache_key in st.session_state.summary_kpi_cache
        and time.time() - st.session_state.summary_kpi_cache[cache_key]["timestamp"]
        < 300
    ):
        summary_data = st.session_state.summary_kpi_cache[cache_key]["data"]
        logging.info("✅ USING CACHED summary data")
    else:
        # Compute summary data - NOW USING SAME LOGIC AS KPI_UTILS
        with st.spinner("🔄 Computing summary statistics using KPI_UTILS logic..."):
            # Extract dataframes
            maternal_events_df = (
                maternal_data.get("events", pd.DataFrame())
                if maternal_data
                else pd.DataFrame()
            )
            newborn_events_df = (
                newborn_data.get("events", pd.DataFrame())
                if newborn_data
                else pd.DataFrame()
            )

            # ✅ FIX: Use events dataframe for counting mothers - SAME AS KPI_UTILS
            # Count unique mothers from events dataframe (same as kpi_utils)
            if not maternal_events_df.empty and "tei_id" in maternal_events_df.columns:
                if facility_uids:
                    filtered_events = maternal_events_df[
                        maternal_events_df["orgUnit"].isin(facility_uids)
                    ]
                else:
                    filtered_events = maternal_events_df
                maternal_tei_count = filtered_events["tei_id"].nunique()
            else:
                maternal_tei_count = 0

            # ✅ FIX: Count unique newborns from events dataframe (consistent approach)
            if not newborn_events_df.empty and "tei_id" in newborn_events_df.columns:
                if facility_uids:
                    filtered_newborn_events = newborn_events_df[
                        newborn_events_df["orgUnit"].isin(facility_uids)
                    ]
                else:
                    filtered_newborn_events = newborn_events_df
                newborn_tei_count = filtered_newborn_events["tei_id"].nunique()
            else:
                newborn_tei_count = 0

            # Get dates from enrollments
            maternal_enrollments_df = normalize_enrollment_dates(
                maternal_data.get("enrollments", pd.DataFrame())
                if maternal_data
                else pd.DataFrame()
            )
            newborn_enrollments_df = normalize_enrollment_dates(
                newborn_data.get("enrollments", pd.DataFrame())
                if newborn_data
                else pd.DataFrame()
            )

            newborn_start_date = get_earliest_date(
                newborn_enrollments_df, "enrollmentDate"
            )
            maternal_start_date = get_earliest_date(
                maternal_enrollments_df, "enrollmentDate"
            )

            # ✅ FIX: Regional comparison - using events dataframe (same as kpi_utils)
            regional_comparison_data = {}
            for region_name, facilities in facilities_by_region.items():
                region_facility_uids = [fac_uid for fac_name, fac_uid in facilities]

                # Count mothers from events (same as kpi_utils)
                if (
                    not maternal_events_df.empty
                    and "tei_id" in maternal_events_df.columns
                    and region_facility_uids
                ):
                    region_maternal_events = maternal_events_df[
                        maternal_events_df["orgUnit"].isin(region_facility_uids)
                    ]
                    maternal_count = region_maternal_events["tei_id"].nunique()
                else:
                    maternal_count = 0

                # Count newborns from events
                if (
                    not newborn_events_df.empty
                    and "tei_id" in newborn_events_df.columns
                    and region_facility_uids
                ):
                    region_newborn_events = newborn_events_df[
                        newborn_events_df["orgUnit"].isin(region_facility_uids)
                    ]
                    newborn_count = region_newborn_events["tei_id"].nunique()
                else:
                    newborn_count = 0

                regional_comparison_data[region_name] = {
                    "mothers": maternal_count,
                    "newborns": newborn_count,
                }

            # Use pre-computed KPIs from kpi_utils
            if (
                st.session_state.get("last_computed_kpis")
                and st.session_state.get("last_computed_facilities") == facility_uids
                and time.time() - st.session_state.get("last_computed_timestamp", 0)
                < 300
            ):
                kpi_data = st.session_state.last_computed_kpis
                logging.info("✅ REUSING pre-computed MATERNAL KPIs from kpi_utils")
            else:
                # Fallback computation using kpi_utils
                if (
                    st.session_state.get("filtered_events") is not None
                    and not st.session_state.filtered_events.empty
                ):
                    kpi_data = compute_kpis(
                        st.session_state.filtered_events, facility_uids
                    )
                elif not maternal_events_df.empty:
                    kpi_data = compute_kpis(maternal_events_df, facility_uids)
                else:
                    kpi_data = {
                        "total_deliveries": maternal_tei_count,
                        "maternal_deaths": 0,
                        "maternal_death_rate": 0.0,
                        "live_births": 0,
                        "stillbirths": 0,
                        "total_births": 0,
                        "stillbirth_rate": 0.0,
                    }
                logging.info("🔄 Computing MATERNAL KPIs for summary using kpi_utils")

            # Extract KPI values from kpi_utils computation
            maternal_indicators = {
                "total_deliveries": kpi_data.get(
                    "total_deliveries", maternal_tei_count
                ),
                "maternal_deaths": kpi_data.get("maternal_deaths", 0),
                "maternal_death_rate": kpi_data.get("maternal_death_rate", 0.0),
                "live_births": kpi_data.get("live_births", 0),
                "stillbirths": kpi_data.get("stillbirths", 0),
                "total_births": kpi_data.get("total_births", 0),
                "stillbirth_rate": kpi_data.get("stillbirth_rate", 0.0),
            }

            # For LBW, use minimal computation from kpi_utils
            lbw_events_df = (
                st.session_state.filtered_events
                if st.session_state.get("filtered_events") is not None
                else maternal_events_df
            )
            lbw_data = compute_lbw_kpi(lbw_events_df, facility_uids)
            maternal_indicators.update(
                {
                    "low_birth_weight_rate": lbw_data.get("lbw_rate", 0.0),
                    "low_birth_weight_count": lbw_data.get("lbw_count", 0),
                }
            )

            # Use pre-computed NEWBORN KPIs
            if (
                st.session_state.get("last_computed_newborn_kpis")
                and time.time()
                - st.session_state.get("last_computed_newborn_timestamp", 0)
                < 300
            ):
                newborn_kpi_data = st.session_state.last_computed_newborn_kpis
                logging.info("✅ REUSING pre-computed NEWBORN KPIs")
            else:
                newborn_kpi_data = {
                    "kmc_coverage_rate": 0.0,
                    "kmc_cases": 0,
                    "total_lbw": 0,
                    "total_newborns": newborn_tei_count,  # Using events count
                    "total_mothers": maternal_tei_count,  # Using events count
                }
                logging.info("🔄 Using placeholder NEWBORN KPIs")

            # Newborn indicators
            newborn_indicators = {
                "total_admitted": newborn_kpi_data.get(
                    "total_newborns", newborn_tei_count
                ),
                "nmr": "N/A",
                "kmc_coverage_rate": newborn_kpi_data.get("kmc_coverage_rate", 0.0),
                "kmc_cases": newborn_kpi_data.get("kmc_cases", 0),
                "total_lbw": newborn_kpi_data.get("total_lbw", 0),
            }

            summary_data = {
                "maternal_indicators": maternal_indicators,
                "newborn_indicators": newborn_indicators,
                "maternal_tei_count": maternal_tei_count,  # Now from events (same as kpi_utils)
                "newborn_tei_count": newborn_tei_count,  # Now from events (consistent)
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

    # ✅ Show quick statistics at the top
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

    # Create newborn table data
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
            f"{maternal_indicators['stillbirth_rate']:.2f} per 1000 births",
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

    # Download button for newborn data
    col_info, col_download = st.columns([3, 1])
    with col_download:
        newborn_data_download = {
            "Start Date": [newborn_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Newborns": [newborn_tei_count],
            "NMR": [f"{newborn_indicators['nmr']}"],
            "Stillbirth Rate": [
                f"{maternal_indicators['stillbirth_rate']:.2f} per 1000 births"
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
            "Maternal Death Rate",
        ],
        "Value": [
            maternal_start_date,
            location_name,
            f"{maternal_tei_count:,}",
            f"{maternal_indicators['total_deliveries']:,}",
            f"{maternal_indicators['maternal_death_rate']:.2f} per 100,000 births",
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
            "Maternal Death Rate": [
                f"{maternal_indicators['maternal_death_rate']:.2f} per 100,000 births"
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
                "📥 Download Regional Data",
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
    """Optimized Maternal Dashboard rendering - Only runs when tab is active"""
    if st.session_state.active_tab != "maternal":
        return

    logging.info("🔄 Maternal dashboard rendering")

    if not maternal_data:
        st.error("No maternal data available")
        return

    # Efficient data extraction
    tei_df = maternal_data.get("tei", pd.DataFrame())
    enrollments_df = maternal_data.get("enrollments", pd.DataFrame())
    events_df = maternal_data.get("events", pd.DataFrame())

    # Normalize dates efficiently
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # Store in session state for quick access
    st.session_state.maternal_events_df = events_df.copy()
    st.session_state.maternal_tei_df = tei_df.copy()

    render_connection_status(events_df, user=user)

    # Calculate totals
    total_facilities = len(facility_mapping)
    total_regions = len(facilities_by_region.keys())

    # Update facility selection
    facility_uids, display_names, comparison_mode = update_facility_selection(
        st.session_state.filter_mode,
        st.session_state.selected_regions,
        st.session_state.selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids and st.session_state.get("facility_filter_applied", False):
        maternal_data = filter_data_by_facilities(maternal_data, facility_uids)
        tei_df = maternal_data.get("tei", pd.DataFrame())
        events_df = maternal_data.get("events", pd.DataFrame())

    # Optimized header rendering
    selected_facilities_count = len(facility_uids)

    header_configs = {
        ("facility", True, 1): (
            f"🤰 Maternal Inpatient Data - {display_names[0]}",
            "1 facility",
        ),
        ("facility", True, "multiple"): (
            "🤰 Maternal Inpatient Data - Multiple Facilities",
            f"{selected_facilities_count} facilities",
        ),
        ("region", True, 1): (
            f"🤰 Maternal Inpatient Data - {display_names[0]} Region",
            f"{sum(len(facilities_by_region.get(region, [])) for region in display_names)} facilities in 1 region",
        ),
        ("region", True, "multiple"): (
            "🤰 Maternal Inpatient Data - Multiple Regions",
            f"{selected_facilities_count} facilities across {len(display_names)} regions",
        ),
    }

    key = (
        comparison_mode,
        "All Facilities" not in display_names,
        1 if len(display_names) == 1 else "multiple",
    )
    header_title, header_subtitle = header_configs.get(
        key,
        (
            f"🤰 Maternal Inpatient Data - {country_name}",
            f"all {total_facilities} facilities",
        ),
    )

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**📊 Displaying data from {header_subtitle}**")

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
            events_df, container=col_ctrl, context="national_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters efficiently
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()
    st.session_state["last_applied_selection"] = True

    # KPI Cards with filtered data
    with kpi_container:
        if filtered_events.empty or "event_date" not in filtered_events.columns:
            progress_container.empty()
            st.markdown(
                '<div class="no-data-warning">⚠️ No Maternal Inpatient Data available for selected filters.</div>',
                unsafe_allow_html=True,
            )
            return

        location_name, location_type = get_location_display_name(
            st.session_state.filter_mode,
            st.session_state.selected_regions,
            st.session_state.selected_facilities,
            country_name,
        )

        user_id = str(user.get("id", user.get("username", "default_user")))

        # ✅ STORE computed KPIs for reuse in summary tab
        kpi_data = render_kpi_cards(filtered_events, location_name, user_id=user_id)

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

        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} - {comparison_mode.title()} Comparison - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            render_comparison_chart(
                kpi_selection=selected_kpi,
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
                f'<div class="section-header" style="margin: 0.3rem 0;">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            render_trend_chart_section(
                selected_kpi,
                filtered_events,
                facility_uids,
                display_names,
                bg_color,
                text_color,
            )

        render_additional_analytics(
            selected_kpi, filtered_events, facility_uids, bg_color, text_color
        )


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
        for region in selected_regions:
            if region in facilities_by_region:
                facility_uids.extend(
                    fac_uid for _, fac_uid in facilities_by_region[region]
                )
        display_names = selected_regions
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
    country_name = user.get("country_name", "Unknown country")

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
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
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
            with st.spinner("🚀 Loading facility data..."):
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

    # ✅ FIX: SINGLE DATA LOADING with PROGRESS INDICATOR
    if not st.session_state.get("data_initialized", False):
        # Show progress indicator for first load
        progress_container = st.empty()
        with progress_container.container():
            st.markdown("---")
            st.markdown("### 🚀 Loading Dashboard Data...")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    """
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
                <h4 style="margin: 0 0 10px 0; color: #1f77b4;">📥 Fetching Data from DHIS2</h4>
                <p style="margin: 5px 0; font-size: 14px;">• Loading maternal inpatient data...</p>
                <p style="margin: 5px 0; font-size: 14px;">• Loading newborn care data...</p>
                <p style="margin: 5px 0; font-size: 14px;">• Processing and caching results...</p>
                <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">⏱️ This may take 3-5 minutes for initial load</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                <div style="text-align: center; padding: 10px;">
                <div style="font-size: 24px;">⏳</div>
                <div style="font-size: 12px; margin-top: 5px;">Loading</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Load the data
        shared_data = get_shared_program_data_optimized(
            user, program_uid_map, show_spinner=False
        )

        # Clear progress indicator
        progress_container.empty()

        st.session_state.data_initialized = True
        st.session_state.cached_shared_data = shared_data
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
            st.markdown("**Select facilities:**")
            updated_facilities = temp_selected_facilities.copy()

            for region_name, facilities in facilities_by_region.items():
                total_count = len(facilities)
                selected_count = sum(
                    1 for fac, _ in facilities if fac in updated_facilities
                )
                icon = (
                    "✅"
                    if selected_count == total_count
                    else "⚠️" if selected_count > 0 else "○"
                )

                with st.expander(
                    f"{icon} {region_name} ({selected_count}/{total_count})",
                    expanded=False,
                ):
                    all_currently_selected = all(
                        fac in updated_facilities for fac, _ in facilities
                    )
                    select_all_checked = st.checkbox(
                        "Select all facilities in this region",
                        value=all_currently_selected,
                        key=f"select_all_fixed_{region_name}",
                    )

                    if select_all_checked:
                        for fac_name, _ in facilities:
                            if fac_name not in updated_facilities:
                                updated_facilities.append(fac_name)
                    else:
                        if all_currently_selected:
                            for fac_name, _ in facilities:
                                if fac_name in updated_facilities:
                                    updated_facilities.remove(fac_name)

                    for fac_name, _ in facilities:
                        is_currently_selected = fac_name in updated_facilities
                        facility_checked = st.checkbox(
                            fac_name,
                            value=is_currently_selected,
                            key=f"fac_fixed_{region_name}_{fac_name}",
                        )
                        if facility_checked and not is_currently_selected:
                            updated_facilities.append(fac_name)
                        elif not facility_checked and is_currently_selected:
                            if fac_name in updated_facilities:
                                updated_facilities.remove(fac_name)

            temp_selected_facilities = updated_facilities

        selection_submitted = st.form_submit_button(
            "✅ Apply", use_container_width=True
        )
        if selection_submitted:
            st.session_state.selected_regions = temp_selected_regions
            st.session_state.selected_facilities = temp_selected_facilities
            st.session_state.selection_applied = True
            st.session_state.facility_filter_applied = True
            st.rerun()

    # ================ COMPACT VIEW MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 3px;">📊 View Mode</p>',
        unsafe_allow_html=True,
    )
    view_mode = st.sidebar.radio(
        "View:",
        ["Normal Trend", "Comparison View"],
        index=0,
        key="view_mode_shared",
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
            logging.info("🔄 Switched to Maternal tab")

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
            logging.info("🔄 Switched to Newborn tab")

        newborn_data = shared_data["newborn"]
        if newborn_data:
            render_newborn_dashboard(
                user,
                program_uid_map.get("Newborn Care Form"),
                country_name,
                facilities_by_region,
                facility_mapping,
                view_mode=view_mode,
                shared_newborn_data=newborn_data,
            )
        else:
            st.error("Newborn data not available")

    with tab3:
        if st.session_state.active_tab != "summary":
            st.session_state.active_tab = "summary"
            logging.info("🔄 Switched to Summary tab")

        render_summary_dashboard_shared(
            user, country_name, facilities_by_region, facility_mapping, shared_data
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

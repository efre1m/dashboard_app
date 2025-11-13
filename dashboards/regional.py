# dashboards/regional.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
from components.kpi_card import render_kpi_cards
from newborns_dashboard.reginal_newborn import render_newborn_dashboard
from utils.data_service import fetch_program_data_for_user
from utils.queries import (
    get_all_programs,
    get_facilities_for_user,
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
@st.cache_data(
    ttl=CACHE_TTL, show_spinner=False, max_entries=5
)  # Increased max_entries for multiple users
def fetch_shared_program_data(user, program_uid):
    """Optimized shared cache for program data - user-specific"""
    if not program_uid:
        return None
    try:
        return fetch_program_data_for_user(user, program_uid)
    except Exception as e:
        logging.error(f"Error fetching data for program {program_uid}: {e}")
        return None


def get_shared_program_data_optimized(user, program_uid_map):
    """Optimized data loading with parallel execution and user-specific caching"""
    maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
    newborn_program_uid = program_uid_map.get("Newborn Care Form")

    # Create user-specific session state keys
    user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    shared_loaded_key = f"shared_data_loaded_{user_key}"
    shared_maternal_key = f"shared_maternal_data_{user_key}"
    shared_newborn_key = f"shared_newborn_data_{user_key}"

    # Initialize session state for shared data - user-specific
    if shared_loaded_key not in st.session_state:
        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None

    # Load data in parallel if not already loaded
    if not st.session_state[shared_loaded_key]:
        with st.spinner("🚀 Loading dashboard data..."):
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
                except concurrent.futures.TimeoutError:
                    logging.error("Data loading timeout")
                    st.error("Data loading timeout. Please try refreshing.")

    return {
        "maternal": st.session_state[shared_maternal_key],
        "newborn": st.session_state[shared_newborn_key],
    }


def clear_shared_cache(user=None):
    """Clear shared data cache - user-specific"""
    if user:
        # Clear specific user cache
        user_key = f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
        shared_loaded_key = f"shared_data_loaded_{user_key}"
        shared_maternal_key = f"shared_maternal_data_{user_key}"
        shared_newborn_key = f"shared_newborn_data_{user_key}"

        st.session_state[shared_loaded_key] = False
        st.session_state[shared_maternal_key] = None
        st.session_state[shared_newborn_key] = None
    else:
        # Clear all user caches (fallback)
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith("shared_data_loaded_")
            or key.startswith("shared_maternal_data_")
            or key.startswith("shared_newborn_data_")
        ]
        for key in keys_to_clear:
            if key.startswith("shared_data_loaded_"):
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    clear_cache()


def initialize_session_state():
    """Optimized session state initialization with user tracking"""
    session_vars = {
        "refresh_trigger": False,
        "selected_facilities": ["All Facilities"],
        "current_facility_uids": [],
        "current_display_names": ["All Facilities"],
        "current_comparison_mode": "facility",
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
        # ADD THESE CRITICAL SESSION STATE VARIABLES:
        "facilities": [],
        "facility_mapping": {},
        "program_uid_map": {},
        "current_user_identifier": None,  # Track current user
        "user_changed": False,  # Flag to detect user changes
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
    else:
        st.session_state.user_changed = False


# Initialize session state at the very beginning
initialize_session_state()


def count_unique_teis_filtered(tei_df, facility_uids, org_unit_column="tei_orgUnit"):
    """Optimized TEI counting"""
    if tei_df.empty or not facility_uids:
        return 0

    # Use vectorized operations for better performance
    if org_unit_column in tei_df.columns:
        filtered_tei = tei_df[tei_df[org_unit_column].isin(facility_uids)]
    else:
        filtered_tei = (
            tei_df[tei_df["orgUnit"].isin(facility_uids)]
            if "orgUnit" in tei_df.columns
            else tei_df
        )

    # Count unique TEIs efficiently
    id_column = (
        "tei_id" if "tei_id" in filtered_tei.columns else "trackedEntityInstance"
    )
    return filtered_tei[id_column].nunique() if id_column in filtered_tei.columns else 0


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


def calculate_maternal_indicators(maternal_events_df, facility_uids):
    """Optimized maternal indicators calculation"""
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

    # Calculate rates
    maternal_death_rate = (
        (maternal_deaths / live_births * 100000) if live_births > 0 else 0.0
    )
    stillbirth_rate = (stillbirths / total_births * 1000) if total_births > 0 else 0.0

    # If lbw_rate is 0 but we have data, calculate it manually
    if low_birth_weight_rate == 0 and total_weighed > 0:
        low_birth_weight_rate = (low_birth_weight_count / total_weighed) * 100

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
    """Optimized newborn indicators calculation"""
    if newborn_events_df.empty:
        return {"total_admitted": 0, "nmr": "N/A"}

    return {"total_admitted": 0, "nmr": "N/A"}


def get_location_display_name(selected_facilities, region_name):
    """Optimized location display name generation"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        return ", ".join(selected_facilities), "Facilities"


def filter_data_by_facilities(data_dict, facility_uids):
    """Filter all dataframes in a data dictionary by facility UIDs"""
    if not data_dict or not facility_uids:
        return data_dict

    filtered_data = {}

    for key, df in data_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Filter based on available organization unit columns
            if "orgUnit" in df.columns:
                filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()
            elif "tei_orgUnit" in df.columns:
                filtered_df = df[df["tei_orgUnit"].isin(facility_uids)].copy()
            else:
                filtered_df = df.copy()  # No filtering possible
            filtered_data[key] = filtered_df
        else:
            filtered_data[key] = df

    return filtered_data


def render_summary_dashboard_shared(
    user, region_name, facility_mapping, selected_facilities, shared_data
):
    """Optimized Summary Dashboard rendering"""
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

    # Use shared data - APPLY FACILITY FILTERING
    maternal_data = shared_data["maternal"]
    newborn_data = shared_data["newborn"]

    if not maternal_data and not newborn_data:
        st.error("No data available for summary dashboard")
        return

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids:
        maternal_data = filter_data_by_facilities(maternal_data, facility_uids)
        newborn_data = filter_data_by_facilities(newborn_data, facility_uids)

    # Extract dataframes efficiently
    maternal_tei_df = (
        maternal_data.get("tei", pd.DataFrame()) if maternal_data else pd.DataFrame()
    )
    maternal_events_df = (
        maternal_data.get("events", pd.DataFrame()) if maternal_data else pd.DataFrame()
    )
    newborn_tei_df = (
        newborn_data.get("tei", pd.DataFrame()) if newborn_data else pd.DataFrame()
    )
    newborn_events_df = (
        newborn_data.get("events", pd.DataFrame()) if newborn_data else pd.DataFrame()
    )

    # Normalize dates efficiently
    newborn_enrollments_df = normalize_enrollment_dates(
        newborn_data.get("enrollments", pd.DataFrame())
        if newborn_data
        else pd.DataFrame()
    )
    maternal_enrollments_df = normalize_enrollment_dates(
        maternal_data.get("enrollments", pd.DataFrame())
        if maternal_data
        else pd.DataFrame()
    )

    # Calculate indicators
    maternal_indicators = calculate_maternal_indicators(
        maternal_events_df, facility_uids
    )
    newborn_indicators = calculate_newborn_indicators(newborn_events_df, facility_uids)

    # Get filtered TEI counts
    maternal_tei_count = count_unique_teis_filtered(
        maternal_tei_df, facility_uids, "tei_orgUnit"
    )
    newborn_tei_count = count_unique_teis_filtered(
        newborn_tei_df, facility_uids, "tei_orgUnit"
    )
    newborn_indicators["total_admitted"] = newborn_tei_count

    # Get earliest dates
    newborn_start_date = get_earliest_date(newborn_enrollments_df, "enrollmentDate")
    maternal_start_date = get_earliest_date(maternal_enrollments_df, "enrollmentDate")

    # Apply optimized table styling with reduced gaps
    st.markdown(
        """
    <style>
    .summary-table-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .summary-table thead tr {
        background: linear-gradient(135deg, #1f77b4, #1668a1);
    }
    .summary-table th {
        color: white;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        border: none;
    }
    .summary-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 13px;
        background-color: white;
    }
    .summary-table tbody tr:last-child td { border-bottom: none; }
    .summary-table tbody tr:hover td { background-color: #f8f9fa; }
    .newborn-table thead tr { background: linear-gradient(135deg, #1f77b4, #1668a1) !important; }
    .maternal-table thead tr { background: linear-gradient(135deg, #2ca02c, #228b22) !important; }
    .summary-table td:first-child { font-weight: 600; color: #666; text-align: center; }
    .summary-table th:first-child { text-align: center; }
    
    /* Reduced gap styling */
    .main-header { margin-bottom: 0.3rem !important; }
    .section-header { margin: 0.3rem 0 !important; }
    .stMarkdown { margin-bottom: 0.2rem !important; }
    .element-container { margin-bottom: 0.3rem !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create overview tables with minimal gaps
    st.markdown("---")

    # Newborn Overview Table
    col_header, col_download = st.columns([3, 1])
    with col_header:
        st.markdown("### 👶 Newborn Care Overview")
    with col_download:
        newborn_data_download = {
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
        newborn_df = pd.DataFrame(newborn_data_download)
        st.download_button(
            "📥 Download Newborn Data",
            data=newborn_df.to_csv(index=False),
            file_name=f"newborn_overview_{location_name.replace(' ', '_').replace(',', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

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
            f"{maternal_indicators['stillbirth_rate']} per 1000 births",
            f"{maternal_indicators['live_births']:,}",
            f"{maternal_indicators['stillbirths']:,}",
            f"{maternal_indicators['total_births']:,}",
            f"{maternal_indicators['low_birth_weight_rate']}%",
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

    # Maternal Overview Table
    st.markdown("---")
    col_header2, col_download2 = st.columns([3, 1])
    with col_header2:
        st.markdown("### 🤰 Maternal Care Overview")
    with col_download2:
        maternal_data_download = {
            "Start Date": [maternal_start_date],
            f"{location_type}": [location_name],
            "Total Admitted Mothers": [maternal_tei_count],
            "Total Deliveries": [maternal_indicators["total_deliveries"]],
            "Maternal Death Rate": [
                f"{maternal_indicators['maternal_death_rate']} per 100,000 births"
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
            f"{maternal_indicators['maternal_death_rate']} per 100,000 births",
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

    # Quick Statistics with optimized styling
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")

    # Create metric cards with reduced spacing
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

    # Facility comparison table
    st.markdown("---")
    st.markdown("### 📊 Mothers & Newborns by facility")

    facility_comparison_data = calculate_facility_comparison_data(
        maternal_tei_df, newborn_tei_df, facility_mapping
    )

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
        col_info, col_download = st.columns([3, 1])
        with col_download:
            download_data = []
            for facility, data in facility_comparison_data.items():
                download_data.append(
                    {
                        "Facility": facility,
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


def calculate_facility_comparison_data(
    maternal_tei_df, newborn_tei_df, facility_mapping
):
    """Optimized facility comparison data calculation"""
    facility_data = {}

    for facility_name, facility_uid in facility_mapping.items():
        maternal_count = count_unique_teis_filtered(
            maternal_tei_df, [facility_uid], "tei_orgUnit"
        )
        newborn_count = count_unique_teis_filtered(
            newborn_tei_df, [facility_uid], "tei_orgUnit"
        )

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
    facility_uids,
    view_mode="Normal Trend",
    facility_mapping=None,
    facility_names=None,
):
    """Optimized Maternal Dashboard rendering"""
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

    # FILTER DATA BY SELECTED FACILITIES
    if facility_uids:
        maternal_data = filter_data_by_facilities(maternal_data, facility_uids)
        # Update dataframes after filtering
        tei_df = maternal_data.get("tei", pd.DataFrame())
        events_df = maternal_data.get("events", pd.DataFrame())

    # Store in session state for quick access
    st.session_state.maternal_events_df = events_df.copy()
    st.session_state.maternal_tei_df = tei_df.copy()

    render_connection_status(events_df, user=user)

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

    # Create containers for better performance
    kpi_container = st.container()

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            events_df, container=col_ctrl, context="regional_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters efficiently
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()
    st.session_state["last_applied_selection"] = True

    # KPI Cards with filtered data
    with kpi_container:
        if filtered_events.empty or "event_date" not in filtered_events.columns:
            st.markdown(
                '<div class="no-data-warning">⚠️ No Maternal Inpatient Data available for selected filters.</div>',
                unsafe_allow_html=True,
            )
            return

        location_name, location_type = get_location_display_name(
            selected_facilities, region_name
        )

        user_id = str(user.get("id", user.get("username", "default_user")))
        render_kpi_cards(filtered_events, location_name, user_id=user_id)

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
                filtered_events=filtered_events,
                comparison_mode="facility",
                display_names=facility_names or selected_facilities,
                facility_uids=facility_uids,
                facilities_by_region=None,
                bg_color=bg_color,
                text_color=text_color,
                is_national=False,
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
                facility_names or selected_facilities,
                bg_color,
                text_color,
            )

        render_additional_analytics(
            selected_kpi,
            filtered_events,
            facility_uids,
            bg_color,
            text_color,
        )


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
        # Clear caches for the new user
        current_user = st.session_state.get("user", {})
        clear_shared_cache(current_user)

    # Load optimized CSS with minimal gaps
    st.markdown(
        """
    <style>
    /* Ultra-compact styling for maximum performance */
    .main-header { 
        font-size: 1.5rem !important; 
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    .section-header {
        font-size: 1.2rem !important;
        margin: 0.2rem 0 !important;
        padding: 0.3rem 0 !important;
    }
    .stMarkdown {
        margin-bottom: 0.1rem !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    .stButton button {
        margin-bottom: 0.2rem !important;
    }
    .stRadio > div {
        padding-top: 0.1rem !important;
        padding-bottom: 0.1rem !important;
    }
    .stForm {
        margin-bottom: 0.3rem !important;
    }
    hr {
        margin: 0.3rem 0 !important;
    }
    .user-info {
        margin-bottom: 0.3rem !important;
        font-size: 0.9rem;
    }
    
    /* Compact sidebar */
    .sidebar .sidebar-content {
        padding: 1rem 0.5rem !important;
    }
    
    /* Metric cards compact */
    .metric-card {
        min-height: 100px !important;
        padding: 12px !important;
        margin: 5px !important;
    }
    .metric-value {
        font-size: 1.5rem !important;
        margin: 5px 0 !important;
    }
    .metric-label {
        font-size: 0.8rem !important;
        margin-bottom: 2px !important;
    }
    .metric-help {
        font-size: 0.65rem !important;
        margin-top: 2px !important;
    }
    
    /* Table compact */
    .summary-table th, .summary-table td {
        padding: 6px 8px !important;
        font-size: 12px !important;
    }
    
    /* Make ALL checkbox labels black and compact */
    .stCheckbox label {
        color: #000000 !important;
        font-size: 0.9rem !important;
    }
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

    # Refresh Data Button - ULTRA COMPACT
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        clear_shared_cache(user)  # Clear specific user cache
        st.session_state.refresh_trigger = not st.session_state.refresh_trigger
        st.session_state.selection_applied = True
        st.rerun()

    # ================ OPTIMIZED DATA LOADING ================
    # Get static data (cached for 1 hour) - FIXED: Use safe access
    if not st.session_state.get("static_data_loaded", False) or st.session_state.get(
        "user_changed", False
    ):
        with st.sidebar:
            with st.spinner("🚀 Loading facility data..."):
                static_data = get_static_data(user)
                # SAFELY initialize session state variables
                st.session_state.facilities = static_data.get("facilities", [])
                st.session_state.facility_mapping = static_data.get(
                    "facility_mapping", {}
                )
                st.session_state.program_uid_map = static_data.get(
                    "program_uid_map", {}
                )
                st.session_state.static_data_loaded = True
                st.session_state.user_changed = False  # Reset user change flag

    # SAFELY access session state variables with defaults
    facilities = st.session_state.get("facilities", [])
    facility_mapping = st.session_state.get("facility_mapping", {})
    program_uid_map = st.session_state.get("program_uid_map", {})

    # Auto-load shared data in background
    shared_data = get_shared_program_data_optimized(user, program_uid_map)

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

    # ================ OPTIMIZED TABS ================
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
        newborn_data = shared_data["newborn"]
        if newborn_data:
            render_newborn_dashboard(
                user=user,
                program_uid=program_uid_map.get("Newborn Care Form"),
                region_name=region_name,
                selected_facilities=selected_facilities,
                facility_uids=facility_uids,
                view_mode=view_mode,
                facility_mapping=facility_mapping,
                facility_names=facility_names,
            )
        else:
            st.error("Newborn data not available")

    with tab3:
        render_summary_dashboard_shared(
            user,
            region_name,
            facility_mapping,
            selected_facilities,
            shared_data,
        )

    with tab4:
        display_odk_dashboard(user)

    with tab5:
        render_data_quality_tracking(user)

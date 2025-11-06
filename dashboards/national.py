# dashboards/national.py
import streamlit as st
import pandas as pd
import logging
import concurrent.futures
import requests
from components.kpi_card import render_kpi_cards
from utils.kpi_lbw import compute_lbw_kpi
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
from utils.status import (
    render_connection_status,
    update_last_sync_time,
    initialize_status_system,
)

from utils.odk_dashboard import display_odk_dashboard
from dashboards.data_quality_tracking import render_data_quality_tracking

# Initialize status system
initialize_status_system()


def initialize_session_state():
    """Initialize all session state variables to prevent AttributeError"""
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
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state at the very beginning
initialize_session_state()

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user, program_uid):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_facilities(user):
    """Cache facility data to avoid repeated API calls"""
    facilities_by_region = get_facilities_grouped_by_region(user)
    facility_mapping = get_facility_mapping_for_user(user)
    return facilities_by_region, facility_mapping


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
        return {"total_admitted": 0, "nmr": "N/A"}  # CHANGED: 0.0 → "N/A"

    # For newborn admitted count, we need to count from filtered TEI data separately
    total_admitted = 0  # Placeholder - will be set from filtered TEI count
    nmr = "N/A"  # CHANGED: 0.0 → "N/A"

    return {"total_admitted": total_admitted, "nmr": nmr}


def get_location_display_name(
    filter_mode, selected_regions, selected_facilities, country_name
):
    """Get the display name for location based on selection"""
    if filter_mode == "All Facilities":
        return country_name, "Country"
    elif filter_mode == "By Region" and selected_regions:
        if len(selected_regions) == 1:
            return selected_regions[0], "Region"
        else:
            # Join multiple regions with comma
            return ", ".join(selected_regions), "Regions"
    elif filter_mode == "By Facility" and selected_facilities:
        if len(selected_facilities) == 1:
            return selected_facilities[0], "Facility"
        else:
            # Join multiple facilities with comma
            return ", ".join(selected_facilities), "Facilities"
    else:
        return country_name, "Country"


def render_summary_dashboard(
    user, country_name, facilities_by_region, facility_mapping
):
    """Render Summary Dashboard with both maternal and newborn overview tables"""

    st.markdown(
        f'<div class="main-header">📊 Summary Dashboard - {country_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Comprehensive overview of maternal and newborn health indicators**")

    # Get programs for UID mapping
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

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

        # ================ REGIONAL COMPARISON TABLE - TRANSPOSED LAYOUT ================
    st.markdown("---")
    st.markdown("### 📊 Mothers & Newborns by region")

    # Calculate regional comparison data
    regional_comparison_data = calculate_regional_comparison_data(
        maternal_tei_df, newborn_tei_df, facilities_by_region, facility_mapping
    )

    # Create the comparison table data
    if regional_comparison_data:
        # Prepare data for the table
        regions = list(regional_comparison_data.keys())

        # Calculate totals
        total_mothers = sum(
            data["mothers"] for data in regional_comparison_data.values()
        )
        total_newborns = sum(
            data["newborns"] for data in regional_comparison_data.values()
        )

        # Create transposed table structure WITHOUT HEADER ROW
        transposed_data = []

        # Add each region as a row
        for i, region in enumerate(regions, 1):
            transposed_data.append(
                {
                    "No": i,
                    "Region Name": region,
                    "Admitted Mothers": f"{regional_comparison_data[region]['mothers']:,}",
                    "Admitted Newborns": f"{regional_comparison_data[region]['newborns']:,}",
                }
            )

        # Add TOTAL row at the bottom
        transposed_data.append(
            {
                "No": "",
                "Region Name": "TOTAL",
                "Admitted Mothers": f"{total_mothers:,}",
                "Admitted Newborns": f"{total_newborns:,}",
            }
        )

        # Convert to DataFrame
        transposed_df = pd.DataFrame(transposed_data)

        # Display the styled table - USING EXACT SAME STYLING AS PREVIOUS TABLES
        st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
        st.markdown(
            transposed_df.style.set_table_attributes(
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
                    # Center align number column
                    {
                        "selector": "td:first-child",
                        "props": [
                            ("text-align", "center"),
                            ("font-weight", "600"),
                            ("color", "#666"),
                        ],
                    },
                    # Center align the numbers columns
                    {
                        "selector": "td:nth-child(3), td:nth-child(4)",
                        "props": [("text-align", "center")],
                    },
                    # Style the TOTAL row - same as previous
                    {
                        "selector": "tbody tr:last-child td",
                        "props": [
                            ("font-weight", "700"),
                            ("background-color", "#f8f9fa"),
                            ("color", "#2c3e50"),
                        ],
                    },
                    # Header styling for number columns
                    {
                        "selector": "th:nth-child(3), th:nth-child(4)",
                        "props": [("text-align", "center")],
                    },
                ]
            )
            .to_html(),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Add info about regions with data
        total_regions_in_country = len(facilities_by_region)
        regions_with_data = len(regional_comparison_data)
        st.markdown(
            f"<small>📋 Showing {regions_with_data} regions with data (out of {total_regions_in_country} total in {country_name})</small>",
            unsafe_allow_html=True,
        )

        # Add download button for regional data
        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col2:
            # Prepare download data
            download_data = []
            for region, data in regional_comparison_data.items():
                download_data.append(
                    {
                        "Region": region,
                        "Admitted Mothers": data["mothers"],
                        "Admitted Newborns": data["newborns"],
                    }
                )

            # Add total row
            download_data.append(
                {
                    "Region": "TOTAL",
                    "Admitted Mothers": total_mothers,
                    "Admitted Newborns": total_newborns,
                }
            )

            download_df = pd.DataFrame(download_data)
            regional_csv = download_df.to_csv(index=False)

            st.download_button(
                "📥 Download Regional Data",
                data=regional_csv,
                file_name=f"regional_comparison_{country_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No regional data available for comparison.")


def calculate_regional_comparison_data(
    maternal_tei_df, newborn_tei_df, facilities_by_region, facility_mapping
):
    """Calculate regional comparison data for mothers and newborns"""
    regional_data = {}

    # Iterate through each region and its facilities
    for region_name, facilities in facilities_by_region.items():
        # Get facility UIDs for this region
        region_facility_uids = [fac_uid for fac_name, fac_uid in facilities]

        # Count mothers in this region
        maternal_count = count_unique_teis_filtered(
            maternal_tei_df, region_facility_uids, "tei_orgUnit"
        )

        # Count newborns in this region
        newborn_count = count_unique_teis_filtered(
            newborn_tei_df, region_facility_uids, "tei_orgUnit"
        )

        regional_data[region_name] = {
            "mothers": maternal_count,
            "newborns": newborn_count,
        }

    return regional_data


def render_maternal_dashboard(
    user,
    program_uid,
    country_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",
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
    events_df = normalize_event_dates(events_df)

    st.session_state.maternal_events_df = events_df.copy()
    st.session_state.maternal_tei_df = tei_df.copy()

    render_connection_status(events_df, user=user)

    # Calculate total counts
    total_facilities = len(facility_mapping)
    total_regions = len(facilities_by_region.keys())

    # ---------------- Update Facility Selection ----------------
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

    # MAIN HEADING with selection summary
    selected_facilities_count = len(facility_uids)

    if comparison_mode == "facility" and "All Facilities" in display_names:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {country_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from all {total_facilities} facilities**")
    elif comparison_mode == "facility" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {display_names[0]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📊 Displaying data from 1 facility**")
    elif comparison_mode == "facility":
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - Multiple Facilities</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities**"
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - {display_names[0]} Region</div>',
            unsafe_allow_html=True,
        )
        region_facilities_count = 0
        for region in display_names:
            if region in facilities_by_region:
                region_facilities_count += len(facilities_by_region[region])
        st.markdown(
            f"**📊 Displaying data from {region_facilities_count} facilities in 1 region**"
        )
    else:
        st.markdown(
            f'<div class="main-header">🤰 Maternal Inpatient Data - Multiple Regions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Displaying data from {selected_facilities_count} facilities across {len(display_names)} regions**"
        )

    # Create a container for KPI cards at the top
    kpi_container = st.container()

    # ---------------- FILTERS IN COLUMN STRUCTURE ----------------
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            events_df, container=col_ctrl, context="national_maternal"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters to get data
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)
    st.session_state["filtered_events"] = filtered_events.copy()
    st.session_state["last_applied_selection"] = True

    # ---------------- KPI CARDS IN THE TOP CONTAINER ----------------
    with kpi_container:
        if filtered_events.empty or "event_date" not in filtered_events.columns:
            st.markdown(
                f'<div class="no-data-warning">⚠️ No Maternal Inpatient Data available for selected filters. KPIs and charts are hidden.</div>',
                unsafe_allow_html=True,
            )
            return

        # Get location display name for KPI cards
        location_name, location_type = get_location_display_name(
            st.session_state.filter_mode,
            st.session_state.selected_regions,
            st.session_state.selected_facilities,
            country_name,
        )

        display_name = location_name
        user_id = str(user.get("id", user.get("username", "default_user")))

        # ✅ KPI CARDS AT THE TOP WITH FILTERED DATA
        render_kpi_cards(
            filtered_events,
            display_name,
            user_id=user_id,
        )

    # ---------------- CHARTS BELOW ----------------
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        # Use KPI tab navigation
        selected_kpi = render_kpi_tab_navigation()

        # Use the passed view_mode parameter
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header">📈 {selected_kpi} - {comparison_mode.title()} Comparison - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

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
                f'<div class="section-header">📈 {selected_kpi} Trend - Maternal Inpatient Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            render_trend_chart_section(
                selected_kpi,
                filtered_events,
                facility_uids,
                display_names,
                bg_color,
                text_color,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        render_additional_analytics(
            selected_kpi,
            filtered_events,
            facility_uids,
            bg_color,
            text_color,
        )


def get_current_selection_summary(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
    total_facilities,
    total_regions,
):
    """Generate selection summary without triggering full re-render"""
    if filter_mode == "All Facilities":
        return (
            f"🏥 ALL FACILITIES SELECTED<br><small>Total: {total_facilities} facilities across {total_regions} regions</small>",
            "selection-counter-all",
        )

    elif filter_mode == "By Region":
        if selected_regions:
            facilities_in_selected_regions = 0
            for region in selected_regions:
                if region in facilities_by_region:
                    facilities_in_selected_regions += len(facilities_by_region[region])

            return (
                f"🌍 REGIONS SELECTED: {len(selected_regions)}/{total_regions}<br><small>Covering {facilities_in_selected_regions} facilities</small>",
                "selection-counter-regions",
            )
        else:
            return (
                f"🌍 SELECT REGIONS<br><small>Choose from {total_regions} regions</small>",
                "selection-counter",
            )

    elif filter_mode == "By Facility":
        if selected_facilities:
            return (
                f"🏢 FACILITIES SELECTED: {len(selected_facilities)}/{total_facilities}<br><small>Across {total_regions} regions</small>",
                "selection-counter-facilities",
            )
        else:
            return (
                f"🏢 SELECT FACILITIES<br><small>Choose from {total_facilities} facilities</small>",
                "selection-counter",
            )

    # Default fallback - should never reach here, but just in case
    return (
        f"🏥 SELECTION MODE<br><small>Choose facilities to display data</small>",
        "selection-counter",
    )


def update_facility_selection(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
):
    """Update facility selection based on current mode and selections"""
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif filter_mode == "By Region" and selected_regions:
        facility_uids, display_names = [], selected_regions
        for region in selected_regions:
            if region in facilities_by_region:
                for fac_name, fac_uid in facilities_by_region[region]:
                    facility_uids.append(fac_uid)
        comparison_mode = "region"
    elif filter_mode == "By Facility" and selected_facilities:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    else:
        # Default fallback - all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    return facility_uids, display_names, comparison_mode


# In national.py - Update the render() function


def render():
    st.set_page_config(
        page_title="National Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Re-initialize session state for safety
    initialize_session_state()

    # Load CSS files
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

    # COMPACT SIDEBAR STYLING
    st.markdown(
        """
    <style>
    /* Make ALL checkbox labels black */
    .stCheckbox label {
        color: #000000 !important;
    }

    /* Make expander content white background */
    .stExpander .streamlit-expanderContent {
        background: white !important;
    }

    /* FIX: Keep region headers visible - prevent them from becoming white */
    .stExpander summary {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    .stExpander summary:hover {
        background: rgba(255, 255, 255, 0.15) !important;
    }

    /* FIX: Prevent region headers from changing color when expanded */
    .stExpander[data-expanded="true"] summary {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    /* FIX: Make sure the region header text stays white in all states */
    .stExpander summary,
    .stExpander summary:focus,
    .stExpander summary:active,
    .stExpander summary:hover,
    .stExpander[data-expanded="true"] summary,
    .stExpander[data-expanded="true"] summary:hover {
        color: white !important;
    }

    /* FIX: Ensure the expander icon/arrow stays visible */
    .stExpander summary svg {
        color: white !important;
        fill: white !important;
    }
    
    /* REDUCE ALL SPACING IN SIDEBAR */
    .sidebar .element-container {
        margin-bottom: 0.2rem !important;
    }
    
    .stButton button {
        margin-bottom: 0.3rem !important;
    }
    
    .stRadio > div {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    
    .stMarkdown {
        margin-bottom: 0.3rem !important;
    }
    
    .stForm {
        margin-bottom: 0.5rem !important;
    }
    
    /* Reduce spacing around dividers */
    hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Compact user info */
    .user-info {
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    country_name = user.get("country_name", "Unknown country")

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🌍 Country: {country_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh Data Button - MINIMAL SPACING
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        clear_cache()
        st.session_state.cached_events_data = None
        st.session_state.cached_enrollments_data = None
        st.session_state.cached_tei_data = None
        st.session_state.refresh_trigger = not st.session_state.refresh_trigger
        st.session_state.selection_applied = True
        st.rerun()

    # ---------------- Get Facilities Data ----------------
    try:
        facilities_by_region, facility_mapping = get_cached_facilities(user)
    except Exception as e:
        st.error(f"⚠️ Error loading facility data: {e}")
        # Set default values to prevent further errors
        facilities_by_region = {}
        facility_mapping = {}

    # Get programs for UID mapping
    programs = get_all_programs()
    program_uid_map = {p["program_name"]: p["program_uid"] for p in programs}

    # ================ FACILITY SELECTION MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 5px;">🏥 Facility Selection Mode</p>',
        unsafe_allow_html=True,
    )

    # Fix the radio index calculation with safe fallback
    radio_options = ["All Facilities", "By Region", "By Facility"]

    # Safely get the index - handle cases where filter_mode might have old values
    try:
        current_index = radio_options.index(st.session_state.filter_mode)
    except ValueError:
        # If filter_mode contains an old value like 'facility', reset to default
        current_index = 0
        st.session_state.filter_mode = "All Facilities"

    new_filter_mode = st.sidebar.radio(
        "Select facilities by:",
        radio_options,
        index=current_index,
        key="mode_radio",
    )

    # Update mode without resetting selections
    if new_filter_mode != st.session_state.filter_mode:
        st.session_state.filter_mode = new_filter_mode
        # Don't reset selections - preserve them across mode changes

    # ---------------- Selection Form ----------------
    with st.sidebar.form("selection_form"):
        temp_selected_regions = st.session_state.selected_regions.copy()
        temp_selected_facilities = st.session_state.selected_facilities.copy()

        if st.session_state.filter_mode == "By Region":
            st.markdown("**🌍 Select Regions**")

            # Multi-select dropdown for regions with facility counts
            region_options = {
                f"{region} ({len(facilities_by_region.get(region, []))} facilities)": region
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
                help="Select one or more regions",
                key="region_multiselect",
            )

            # Convert back to region names
            temp_selected_regions = [
                region_options[label] for label in selected_region_labels
            ]

        elif st.session_state.filter_mode == "By Facility":
            st.markdown("**🏢 Select Facilities (grouped by region)**")

            for region_name, facilities in facilities_by_region.items():
                total_count = len(facilities)
                selected_count = sum(
                    1 for fac, _ in facilities if fac in temp_selected_facilities
                )

                # Determine selection status
                if selected_count == 0:
                    header_class = "region-header-none-selected"
                    icon = "○"
                elif selected_count == total_count:
                    header_class = "region-header-fully-selected"
                    icon = "✅"
                else:
                    header_class = "region-header-partially-selected"
                    icon = "⚠️"

                with st.expander(
                    f"{icon} {region_name} ({selected_count}/{total_count} selected)",
                    expanded=False,
                ):
                    # Select all checkbox
                    all_selected_in_region = all(
                        fac in temp_selected_facilities for fac, _ in facilities
                    )
                    select_all_box = st.checkbox(
                        f"Select all in {region_name}",
                        value=all_selected_in_region,
                        key=f"select_all_{region_name}",
                    )

                    # Handle select all logic
                    if select_all_box and not all_selected_in_region:
                        for fac_name, _ in facilities:
                            if fac_name not in temp_selected_facilities:
                                temp_selected_facilities.append(fac_name)
                    elif not select_all_box and all_selected_in_region:
                        for fac_name, _ in facilities:
                            if fac_name in temp_selected_facilities:
                                temp_selected_facilities.remove(fac_name)

                    # Individual facility checkboxes
                    for fac_name, _ in facilities:
                        fac_checked = fac_name in temp_selected_facilities
                        fac_checked = st.checkbox(
                            fac_name,
                            value=fac_checked,
                            key=f"fac_{region_name}_{fac_name}",
                        )
                        if fac_checked and fac_name not in temp_selected_facilities:
                            temp_selected_facilities.append(fac_name)
                        elif not fac_checked and fac_name in temp_selected_facilities:
                            temp_selected_facilities.remove(fac_name)

        else:  # All Facilities mode
            st.markdown("**🏥 All Facilities Mode**")

        selection_submitted = st.form_submit_button("✅ Apply Selection")

        if selection_submitted:
            # Update selections and trigger data display
            st.session_state.selected_regions = temp_selected_regions
            st.session_state.selected_facilities = temp_selected_facilities
            st.session_state.selection_applied = True
            st.rerun()

    # ================ DASHBOARD VIEW MODE ================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 5px;">📊 Dashboard View Mode</p>',
        unsafe_allow_html=True,
    )

    # Shared View Mode for both Maternal and Newborn dashboards
    view_mode = st.sidebar.radio(
        "Select how to view data:",
        ["Normal Trend", "Comparison View"],
        index=0,
        help="Normal Trend: Single trend line | Comparison View: Compare multiple facilities/regions",
        key="view_mode_shared",
    )

    # CREATE PROFESSIONAL TABS IN MAIN AREA
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🤰 **Maternal Inpatient Data**",
            "👶 **Newborn Inpatient Data**",
            "📊 **Summary Dashboard**",
            "📋 **Integrated Mentorship Data**",
            "🔍 **Data Quality Tracking**",
        ]
    )

    with tab1:
        # GROUP 1: Maternal Inpatient Data Content
        maternal_program_uid = program_uid_map.get("Maternal Inpatient Data")
        if maternal_program_uid:
            render_maternal_dashboard(
                user,
                maternal_program_uid,
                country_name,
                facilities_by_region,
                facility_mapping,
                view_mode=view_mode,  # Pass the shared view mode
            )
        else:
            st.error("Maternal Inpatient Data program not found")

    with tab2:
        # GROUP 2: Newborn Care Form Content
        newborn_program_uid = program_uid_map.get("Newborn Care Form")
        if newborn_program_uid:
            render_newborn_dashboard(
                user,
                newborn_program_uid,
                country_name,
                facilities_by_region,
                facility_mapping,
                view_mode=view_mode,  # Pass the shared view mode
            )
        else:
            st.error("Newborn Care Form program not found")

    with tab3:
        # GROUP 3: Summary Dashboard Content
        render_summary_dashboard(
            user,
            country_name,
            facilities_by_region,
            facility_mapping,
        )
    with tab4:
        # NEW: ODK Forms Dashboard Tab
        display_odk_dashboard(user)

    with tab5:
        # NEW: Data Quality Tracking Tab
        render_data_quality_tracking(user)

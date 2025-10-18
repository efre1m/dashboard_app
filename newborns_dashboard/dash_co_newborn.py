# common_dash_neonatal.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import auto_text_color
from newborns_dashboard.kmc_coverage import (
    compute_kmc_kpi,
    render_kmc_trend_chart,
    render_kmc_facility_comparison_chart,
    render_kmc_region_comparison_chart,
)

# KPI mapping for KMC coverage only
KPI_MAPPING = {
    "LBW KMC Coverage (%)": {
        "title": "LBW KMC Coverage (%)",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total LBW Newborns",
    },
}

# Only KMC coverage KPI
KPI_OPTIONS = ["LBW KMC Coverage (%)"]

# KPI Grouping for Tab Navigation
KPI_GROUPS = {
    "Newborn Care": [
        "LBW KMC Coverage (%)",
    ],
}


def get_text_color(bg_color):
    """Get auto text color for background"""
    return auto_text_color(bg_color)


def get_kpi_config(kpi_selection):
    """Get KPI configuration"""
    return KPI_MAPPING.get(kpi_selection, {})


def render_kpi_tab_navigation():
    """Render professional tab navigation for Neonatal KPI selection"""

    # Custom CSS for professional tab styling
    st.markdown(
        """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1a6790 !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #1668a1 !important;
        color: white !important;
        border-color: #145a8c !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border-color: #dee2e6 !important;
        font-weight: 500 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e9ecef !important;
        color: #495057 !important;
        border-color: #ced4da !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for KPI selection
    if "selected_kpi_neonatal" not in st.session_state:
        st.session_state.selected_kpi_neonatal = "LBW KMC Coverage (%)"

    # Create single tab for Newborn Care
    tab1 = st.tabs(["👶 **Newborn Care**"])[0]

    selected_kpi = st.session_state.selected_kpi_neonatal

    with tab1:
        # Only KMC Coverage KPI
        (col1,) = st.columns(1)

        with col1:
            if st.button(
                "📊 KMC Coverage",
                key="kmc_btn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "LBW KMC Coverage (%)" else "secondary"
                ),
            ):
                selected_kpi = "LBW KMC Coverage (%)"

    # Update session state with final selection
    if selected_kpi != st.session_state.selected_kpi_neonatal:
        st.session_state.selected_kpi_neonatal = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi_neonatal


def render_trend_chart_section(
    kpi_selection, filtered_events, facility_uids, display_names, bg_color, text_color
):
    """Render the trend chart based on KPI selection - ONLY LINE CHART"""

    if kpi_selection == "LBW KMC Coverage (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            kmc_data = compute_kmc_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": kmc_data["kmc_rate"],
                    "KMC Cases": kmc_data["kmc_count"],
                    "Total LBW Newborns": kmc_data["total_lbw"],
                }
            )

        group = pd.DataFrame(period_data)
        render_kmc_trend_chart(
            group,
            "period_display",
            "value",
            "LBW KMC Coverage (%)",
            bg_color,
            text_color,
            display_names,
            "KMC Cases",
            "Total LBW Newborns",
            facility_uids,
        )


def render_comparison_chart(
    kpi_selection,
    filtered_events,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    bg_color,
    text_color,
    is_national=False,
):
    """Render comparison charts for both national and regional views - ONLY LINE CHARTS"""

    kpi_config = get_kpi_config(kpi_selection)

    if comparison_mode == "facility":
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%)",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )

    else:  # region comparison (only for national)
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%)",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )


def render_additional_analytics(
    kpi_selection, filtered_events, facility_uids, bg_color, text_color
):
    """NO ADDITIONAL ANALYTICS - as requested"""
    pass


def normalize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a single datetime column 'event_date' exists and is timezone-naive.
    Handles:
      - eventDate like '2025-07-25T00:00:00.000'
      - event_date like '7/25/2025'
    """
    if df.empty:
        return df

    df = df.copy()

    # Parse ISO 'eventDate' if present
    if "eventDate" in df.columns:
        # pandas can parse ISO 8601 with milliseconds without explicit format
        iso_parsed = pd.to_datetime(df["eventDate"], errors="coerce")
    else:
        iso_parsed = pd.Series(pd.NaT, index=df.index)

    # Parse US 'event_date' (m/d/Y) if present
    if "event_date" in df.columns:
        us_parsed = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    else:
        us_parsed = pd.Series(pd.NaT, index=df.index)

    # Prefer ISO if available, else fallback to US
    df["event_date"] = iso_parsed.where(iso_parsed.notna(), us_parsed)

    # Final safety: coerce any str leftovers
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    return df


def normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure enrollmentDate is datetime from '7/25/2025' format."""
    if df.empty or "enrollmentDate" not in df.columns:
        return df
    df = df.copy()
    df["enrollmentDate"] = pd.to_datetime(
        df["enrollmentDate"], format="%m/%d/%Y", errors="coerce"
    )
    return df


def render_simple_filter_controls(events_df, container=None, context="neonatal"):
    """Simple filter controls without KPI selection (KPI selection moved to tabs)"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix based on context
    key_suffix = f"_{context}"

    # NOTE: KPI Selection removed - now handled by tab navigation

    # Time Period
    filters["quick_range"] = container.selectbox(
        "📅 Time Period",
        [
            "Custom Range",
            "Today",
            "This Week",
            "Last Week",
            "This Month",
            "Last Month",
            "This Year",
            "Last Year",
        ],
        index=0,
        key=f"quick_range{key_suffix}",  # Unique key
    )

    # Get dates from dataframe
    min_date, max_date = _get_simple_date_range(events_df)

    # Handle Custom Range vs Predefined Ranges
    if filters["quick_range"] == "Custom Range":
        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"start_date{key_suffix}",  # Unique key
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"end_date{key_suffix}",  # Unique key
            )
    else:
        # For predefined ranges
        _df_for_dates = (
            events_df[["event_date"]].copy()
            if not events_df.empty and "event_date" in events_df.columns
            else pd.DataFrame()
        )
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )
    # Force default to "Monthly" if present, otherwise fallback to first option
    if "Monthly" in available_aggregations:
        default_index = available_aggregations.index("Monthly")
    else:
        default_index = 0

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",  # Unique key
    )

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"  # Unique key
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add a placeholder for kpi_selection to maintain compatibility
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi_neonatal", "LBW KMC Coverage (%)"
    )

    return filters


def _get_simple_date_range(events_df):
    """Get min/max dates from dataframe"""
    import datetime

    if not events_df.empty and "event_date" in events_df.columns:
        valid_dates = events_df["event_date"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            if hasattr(min_date, "date"):
                min_date = min_date.date()
            if hasattr(max_date, "date"):
                max_date = max_date.date()
            return min_date, max_date

    # Fallback to current date
    today = datetime.date.today()
    return today, today


def apply_simple_filters(events_df, filters, facility_uids=None):
    """Apply simple filters to events dataframe"""
    if events_df.empty:
        return events_df

    df = events_df.copy()

    # Apply date filters
    start_datetime = pd.to_datetime(filters["start_date"])
    end_datetime = pd.to_datetime(filters["end_date"])

    df = df[
        (df["event_date"] >= start_datetime) & (df["event_date"] <= end_datetime)
    ].copy()

    # Apply facility filter if provided
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Assign period
    df = assign_period(df, "event_date", filters["period_label"])

    return df

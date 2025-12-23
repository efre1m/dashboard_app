# dash_co_newborn.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import auto_text_color

# Import KPI computation functions - UPDATED to work with patient-level data
from newborns_dashboard.kmc_coverage import (
    compute_kmc_kpi,
    render_kmc_trend_chart,
    render_kmc_facility_comparison_chart,
    render_kmc_region_comparison_chart,
)

# Import other KPI modules
from newborns_dashboard.kpi_cpap import (
    compute_cpap_kpi,
    compute_cpap_general_kpi,
    compute_cpap_prophylactic_kpi,
    render_cpap_trend_chart,
    render_cpap_general_trend_chart,
    render_cpap_prophylactic_trend_chart,
    render_cpap_facility_comparison_chart,
    render_cpap_general_facility_comparison_chart,
    render_cpap_prophylactic_facility_comparison_chart,
    render_cpap_region_comparison_chart,
    render_cpap_general_region_comparison_chart,
    render_cpap_prophylactic_region_comparison_chart,
)

# Import all other KPI modules (keeping them as they should work with patient data)
from newborns_dashboard.kpi_hypothermia import (
    compute_hypothermia_kpi,
    render_hypothermia_trend_chart,
    render_hypothermia_facility_comparison_chart,
    render_hypothermia_region_comparison_chart,
)

from newborns_dashboard.kpi_hypo_after_adm import (
    compute_hypothermia_after_admission_kpi,
    render_hypothermia_after_admission_trend_chart,
    render_hypothermia_after_admission_facility_comparison_chart,
    render_hypothermia_after_admission_region_comparison_chart,
)

from newborns_dashboard.kpi_kmc_1000_2499 import (
    compute_kmc_1000_1999_kpi,
    compute_kmc_2000_2499_kpi,
    render_kmc_both_ranges_trend_chart,
    render_kmc_both_ranges_facility_comparison_chart,
    render_kmc_both_ranges_region_comparison_chart,
)

from newborns_dashboard.kpi_inborn import (
    compute_inborn_kpi,
    render_inborn_trend_chart,
    render_inborn_facility_comparison_chart,
    render_inborn_region_comparison_chart,
)

from newborns_dashboard.kpi_antibiotic import (
    compute_antibiotics_kpi,
    render_antibiotics_trend_chart,
    render_antibiotics_facility_comparison_chart,
    render_antibiotics_region_comparison_chart,
)

from newborns_dashboard.kpi_culture_done import (
    compute_culture_done_kpi,
    render_culture_done_trend_chart,
    render_culture_done_facility_comparison_chart,
    render_culture_done_region_comparison_chart,
)

from newborns_dashboard.kpi_culture_sepsis import (
    compute_culture_done_sepsis_kpi,
    render_culture_done_sepsis_trend_chart,
    render_culture_done_sepsis_facility_comparison_chart,
    render_culture_done_sepsis_region_comparison_chart,
)

from newborns_dashboard.kpi_culture_result_recorded import (
    compute_culture_result_recorded_kpi,
    render_culture_result_recorded_trend_chart,
    render_culture_result_recorded_facility_comparison_chart,
    render_culture_result_recorded_region_comparison_chart,
)

from newborns_dashboard.kpi_nmr import (
    compute_nmr_kpi,
    render_nmr_trend_chart,
    render_nmr_facility_comparison_chart,
    render_nmr_region_comparison_chart,
)

from newborns_dashboard.kpi_newborn_bw import (
    compute_newborn_bw_kpi,
    render_newborn_bw_trend_chart,
    render_newborn_bw_facility_comparison_chart,
    render_newborn_bw_region_comparison_chart,
)

from newborns_dashboard.kpi_in_out_hypo import (
    compute_inborn_outborn_hypothermia_kpi,
    render_inborn_outborn_hypothermia_trend_chart,
    render_inborn_outborn_hypothermia_comparison_chart,
    render_inborn_outborn_hypothermia_summary,
)

# KPI mapping
KPI_MAPPING = {
    "LBW KMC Coverage (%)": {
        "title": "LBW KMC Coverage (%)",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total LBW Newborns",
    },
    "KMC Coverage by Birth Weight Range": {
        "title": "KMC Coverage by Birth Weight Range",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total Newborns by Weight Range",
    },
    "CPAP Coverage for RDS (%)": {
        "title": "CPAP Coverage for RDS (%)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total RDS Newborns",
    },
    "General CPAP Coverage (%)": {
        "title": "General CPAP Coverage (%)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total Admissions",
    },
    "Prophylactic CPAP Coverage (%)": {
        "title": "Prophylactic CPAP Coverage (%)",
        "numerator_name": "Prophylactic CPAP Cases",
        "denominator_name": "Total Newborns (1000-2499g)",
    },
    "Hypothermia on Admission (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admissions",
    },
    "Hypothermia After Admission (%)": {
        "title": "Hypothermia After Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admissions",
    },
    "Hypothermia Inborn/Outborn": {
        "title": "Hypothermia Inborn/Outborn",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Babies by Birth Location",
    },
    "Inborn Babies (%)": {
        "title": "Inborn Babies (%)",
        "numerator_name": "Inborn Cases",
        "denominator_name": "Total Admissions",
    },
    "Antibiotics for Clinical Sepsis (%)": {
        "title": "Antibiotics for Clinical Sepsis (%)",
        "numerator_name": "Antibiotics Cases",
        "denominator_name": "Probable Sepsis Cases",
    },
    "Culture Done for Babies on Antibiotics (%)": {
        "title": "Culture Done for Babies on Antibiotics (%)",
        "numerator_name": "Culture Done Cases",
        "denominator_name": "Total Babies on Antibiotics",
    },
    "Culture Done for Babies with Clinical Sepsis (%)": {
        "title": "Culture Done for Babies with Clinical Sepsis (%)",
        "numerator_name": "Culture Done Cases",
        "denominator_name": "Probable Sepsis Cases",
    },
    "Culture Result Recorded (%)": {
        "title": "Culture Result Recorded (%)",
        "numerator_name": "Culture Result Recorded Cases",
        "denominator_name": "Total Culture Done Cases",
    },
    "Neonatal Mortality Rate (%)": {
        "title": "Neonatal Mortality Rate (%)",
        "numerator_name": "Dead Cases",
        "denominator_name": "Total Admissions",
    },
    "Newborn Birth Weight Distribution": {
        "title": "Newborn Birth Weight Distribution",
        "numerator_name": "Birth Weight Cases",
        "denominator_name": "Total Admissions",
    },
}

# KPI options
KPI_OPTIONS = [
    "LBW KMC Coverage (%)",
    "KMC Coverage by Birth Weight Range",
    "CPAP Coverage for RDS (%)",
    "General CPAP Coverage (%)",
    "Prophylactic CPAP Coverage (%)",
    "Antibiotics for Clinical Sepsis (%)",
    "Culture Done for Babies on Antibiotics (%)",
    "Culture Done for Babies with Clinical Sepsis (%)",
    "Culture Result Recorded (%)",
    "Hypothermia on Admission (%)",
    "Hypothermia After Admission (%)",
    "Hypothermia Inborn/Outborn",
    "Inborn Babies (%)",
    "Newborn Birth Weight Distribution",
    "Neonatal Mortality Rate (%)",
]

# KPI Groups for Tab Navigation
KPI_GROUPS = {
    "Newborn Care": [
        "LBW KMC Coverage (%)",
        "KMC Coverage by Birth Weight Range",
        "CPAP Coverage for RDS (%)",
        "General CPAP Coverage (%)",
        "Prophylactic CPAP Coverage (%)",
        "Antibiotics for Clinical Sepsis (%)",
        "Culture Done for Babies on Antibiotics (%)",
        "Culture Done for Babies with Clinical Sepsis (%)",
        "Culture Result Recorded (%)",
    ],
    "Admission Assessment": [
        "Hypothermia on Admission (%)",
        "Hypothermia After Admission (%)",
        "Hypothermia Inborn/Outborn",
        "Inborn Babies (%)",
        "Newborn Birth Weight Distribution",
    ],
    "Newborn Outcomes": [
        "Neonatal Mortality Rate (%)",
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

    # Custom CSS for button styling
    st.markdown(
        """
    <style>
    div.stButton > button {
        padding: 0.2rem 0.6rem !important;
        font-size: 0.8rem !important;
        height: auto !important;
        min-height: 1.8rem !important;
        margin: 0.05rem !important;
    }
    
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

    # ✅ UNIQUE session state key for newborn dashboard
    if "selected_kpi_NEWBORN_DASHBOARD" not in st.session_state:
        st.session_state.selected_kpi_NEWBORN_DASHBOARD = "LBW KMC Coverage (%)"

    # Create tabs for all groups
    tab1, tab2, tab3 = st.tabs(
        [
            "👶 **Newborn Care**",
            "🩺 **Admission Assessment**",
            "📊 **Newborn Outcomes**",
        ]
    )

    selected_kpi = st.session_state.selected_kpi_NEWBORN_DASHBOARD

    with tab1:
        # Newborn Care KPIs - First row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button(
                "LBW KMC Coverage",
                key="kmc_lbw_btn_newborn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "LBW KMC Coverage (%)" else "secondary"
                ),
            ):
                selected_kpi = "LBW KMC Coverage (%)"

        with col2:
            if st.button(
                "KMC by BW Range",
                key="kmc_both_ranges_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "KMC Coverage by Birth Weight Range"
                    else "secondary"
                ),
            ):
                selected_kpi = "KMC Coverage by Birth Weight Range"

        with col3:
            if st.button(
                "CPAP for RDS",
                key="cpap_rds_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "CPAP Coverage for RDS (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "CPAP Coverage for RDS (%)"

        with col4:
            if st.button(
                "General CPAP",
                key="cpap_general_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "General CPAP Coverage (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "General CPAP Coverage (%)"

        with col5:
            if st.button(
                "Prophylactic CPAP",
                key="cpap_prophylactic_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Prophylactic CPAP Coverage (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Prophylactic CPAP Coverage (%)"

        # Second row
        col6, col7, col8, col9 = st.columns(4)

        with col6:
            if st.button(
                "Antibiotics for Sepsis",
                key="antibiotics_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Antibiotics for Clinical Sepsis (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Antibiotics for Clinical Sepsis (%)"

        with col7:
            if st.button(
                "Culture Done",
                key="culture_done_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Culture Done for Babies on Antibiotics (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Culture Done for Babies on Antibiotics (%)"

        with col8:
            if st.button(
                "Culture for Sepsis",
                key="culture_done_sepsis_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi
                    == "Culture Done for Babies with Clinical Sepsis (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Culture Done for Babies with Clinical Sepsis (%)"

        with col9:
            if st.button(
                "Culture Result",
                key="culture_result_recorded_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Culture Result Recorded (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Culture Result Recorded (%)"

    with tab2:
        # Admission Assessment KPIs
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "Hypothermia on Admission",
                key="hypothermia_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia on Admission (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia on Admission (%)"

        with col2:
            if st.button(
                "Hypothermia After Admission",
                key="hypothermia_after_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia After Admission (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia After Admission (%)"

        with col3:
            if st.button(
                "Hypothermia Inborn/Outborn",
                key="hypo_in_out_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia Inborn/Outborn"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia Inborn/Outborn"

        with col4:
            if st.button(
                "Inborn Babies",
                key="inborn_btn_newborn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "Inborn Babies (%)" else "secondary"
                ),
            ):
                selected_kpi = "Inborn Babies (%)"

        # Birth Weight button
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button(
                "Birth Weight",
                key="bw_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Newborn Birth Weight Distribution"
                    else "secondary"
                ),
            ):
                selected_kpi = "Newborn Birth Weight Distribution"

    with tab3:
        # Newborn Outcomes KPIs
        (col1,) = st.columns(1)

        with col1:
            if st.button(
                "Neonatal Mortality",
                key="nmr_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Neonatal Mortality Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Neonatal Mortality Rate (%)"

    # Update session state if changed
    if selected_kpi != st.session_state.selected_kpi_NEWBORN_DASHBOARD:
        st.session_state.selected_kpi_NEWBORN_DASHBOARD = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi_NEWBORN_DASHBOARD


def get_event_date_from_patient_df(patient_df):
    """✅ Extract event date from patient-level dataframe"""
    if patient_df.empty:
        return pd.Series([])

    # Look for date columns
    date_cols = [
        col
        for col in patient_df.columns
        if "date" in col.lower() or "Date" in col or "period" in col.lower()
    ]

    if date_cols:
        # Use the first date column found
        date_col = date_cols[0]
        return pd.to_datetime(patient_df[date_col], errors="coerce")
    else:
        # No date column found, return empty series
        return pd.Series([pd.NaT] * len(patient_df))


def normalize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ UPDATED for patient-level data
    Extract event date from patient-level columns
    """
    if df.empty:
        return df

    df = df.copy()

    # For patient-level data, we need to find date columns
    if "event_date" not in df.columns:
        # Look for date columns in the patient data
        date_series = get_event_date_from_patient_df(df)
        df["event_date"] = date_series

        # Create placeholder period columns
        df["period"] = "Unknown"
        df["period_display"] = "Unknown"
        df["period_sort"] = "999999"

    return df


def normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """✅ COMPATIBILITY FUNCTION - Patient-level data doesn't have enrollment dates"""
    if df.empty:
        return df

    df = df.copy()

    if "enrollmentDate" not in df.columns:
        df["enrollmentDate"] = pd.NaT

    return df


def apply_simple_filters(events_df, filters, facility_uids=None):
    """✅ OPTIMIZED: Apply simple filters to patient-level data"""
    if events_df.empty:
        return pd.DataFrame()

    # Start with all data
    filtered_df = events_df.copy()

    # Filter by date if event_date exists
    if "event_date" in filtered_df.columns and filters:
        start_datetime = pd.to_datetime(filters["start_date"])
        end_datetime = pd.to_datetime(filters["end_date"])

        date_mask = (filtered_df["event_date"] >= start_datetime) & (
            filtered_df["event_date"] <= end_datetime
        )
        filtered_df = filtered_df[date_mask].copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    # Assign period for grouping if period_label is provided
    if filters and "period_label" in filters and "event_date" in filtered_df.columns:
        filtered_df = assign_period(filtered_df, "event_date", filters["period_label"])

    return filtered_df


def render_simple_filter_controls(patient_df, container=None, context="newborn"):
    """Render simple filter controls for patient-level data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix based on context
    key_suffix = f"_{context}"

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
        key=f"quick_range{key_suffix}",
    )

    # Get dates from dataframe
    min_date, max_date = _get_simple_date_range(patient_df)

    # Handle Custom Range vs Predefined Ranges
    if filters["quick_range"] == "Custom Range":
        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"start_date{key_suffix}",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"end_date{key_suffix}",
            )
    else:
        # For predefined ranges
        _df_for_dates = patient_df.copy() if not patient_df.empty else pd.DataFrame()
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )
    if "Monthly" in available_aggregations:
        default_index = available_aggregations.index("Monthly")
    else:
        default_index = 0

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",
    )

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # KPI selection from session state
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi_NEWBORN_DASHBOARD", "LBW KMC Coverage (%)"
    )

    return filters


def _get_simple_date_range(df):
    """Get min/max dates from patient-level dataframe"""
    import datetime

    if not df.empty:
        # Try to find a date column
        date_series = get_event_date_from_patient_df(df)
        valid_dates = date_series.dropna()
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


def render_trend_chart_section(
    selected_kpi,
    filtered_data,  # ✅ Can be either patient_df or events_df
    facility_uids,
    facility_names,
    bg_color,
    text_color,
    tei_df=None,
):
    """Render the trend chart based on KPI selection using patient-level data"""

    # ✅ Determine if we have patient-level data
    is_patient_level = (
        "tei_id" in filtered_data.columns and "orgUnit" in filtered_data.columns
    )

    if is_patient_level:
        # Convert patient data to events format for chart functions that expect it
        # Create a simple events-like structure
        events_list = []
        for idx, row in filtered_data.iterrows():
            events_list.append(
                {
                    "tei_id": row.get("tei_id", ""),
                    "orgUnit": row.get("orgUnit", ""),
                    "event_date": row.get("event_date", pd.NaT),
                    "period": row.get("period", "Unknown"),
                    "period_display": row.get("period_display", "Unknown"),
                }
            )
        events_df = pd.DataFrame(events_list)
    else:
        events_df = filtered_data

    if selected_kpi == "LBW KMC Coverage (%)":
        render_kmc_trend_chart(
            events_df,
            "period_display",
            "LBW KMC Coverage (%)",
            bg_color,
            text_color,
            facility_names,
            "KMC Cases",
            "Total LBW Newborns",
            facility_uids,
        )

    elif selected_kpi == "KMC Coverage by Birth Weight Range":
        render_kmc_both_ranges_trend_chart(
            events_df,
            "period_display",
            "KMC Coverage by Birth Weight Range",
            bg_color,
            text_color,
            facility_names,
            facility_uids=facility_uids,
        )

    elif selected_kpi == "CPAP Coverage for RDS (%)":
        render_cpap_trend_chart(
            events_df,
            "period_display",
            "CPAP Coverage for RDS (%)",
            bg_color,
            text_color,
            facility_names,
            "CPAP Cases",
            "Total RDS Newborns",
            facility_uids,
        )

    elif selected_kpi == "General CPAP Coverage (%)":
        render_cpap_general_trend_chart(
            events_df,
            "period_display",
            "General CPAP Coverage Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Prophylactic CPAP Coverage (%)":
        render_cpap_prophylactic_trend_chart(
            events_df,
            "period_display",
            "Prophylactic CPAP Coverage Trend",
            bg_color,
            text_color,
            facility_names,
            "Prophylactic CPAP Cases",
            "Total Newborns (1000-2499g)",
            facility_uids,
        )

    elif selected_kpi == "Antibiotics for Clinical Sepsis (%)":
        render_antibiotics_trend_chart(
            events_df,
            "period_display",
            "Antibiotics for Clinical Sepsis Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Culture Done for Babies on Antibiotics (%)":
        render_culture_done_trend_chart(
            events_df,
            "period_display",
            "Culture Done for Babies on Antibiotics Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Culture Done for Babies with Clinical Sepsis (%)":
        render_culture_done_sepsis_trend_chart(
            events_df,
            "period_display",
            "Culture Done for Babies with Clinical Sepsis Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Culture Result Recorded (%)":
        render_culture_result_recorded_trend_chart(
            events_df,
            "period_display",
            "Blood Culture Result Recorded Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Hypothermia on Admission (%)":
        render_hypothermia_trend_chart(
            events_df,
            "period_display",
            "Hypothermia on Admission (%)",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Hypothermia After Admission (%)":
        render_hypothermia_after_admission_trend_chart(
            events_df,
            "period_display",
            "Hypothermia After Admission (%)",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Hypothermia Inborn/Outborn":
        render_inborn_outborn_hypothermia_trend_chart(
            events_df,
            "period_display",
            "Hypothermia at Admission by Birth Location",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Inborn Babies (%)":
        render_inborn_trend_chart(
            events_df,
            "period_display",
            "Inborn Babies Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif selected_kpi == "Newborn Birth Weight Distribution":
        render_newborn_bw_trend_chart(
            events_df,
            "period_display",
            "Newborn Birth Weight Distribution Trend",
            bg_color,
            text_color,
            facility_names=facility_names,
            facility_uids=facility_uids,
        )

    elif selected_kpi == "Neonatal Mortality Rate (%)":
        render_nmr_trend_chart(
            events_df,
            "period_display",
            "Neonatal Mortality Rate Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
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
    tei_df=None,
):
    """Render comparison charts for facility comparison using patient-level data"""

    if comparison_mode == "facility":
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="LBW KMC Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )
        elif kpi_selection == "KMC Coverage by Birth Weight Range":
            render_kmc_both_ranges_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="KMC Coverage by Birth Weight Range - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
            )
        elif kpi_selection == "CPAP Coverage for RDS (%)":
            render_cpap_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="CPAP Coverage for RDS (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="CPAP Cases",
                denominator_name="Total RDS Newborns",
            )
        elif kpi_selection == "General CPAP Coverage (%)":
            render_cpap_general_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="General CPAP Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Prophylactic CPAP Coverage (%)":
            render_cpap_prophylactic_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Prophylactic CPAP Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="Prophylactic CPAP Cases",
                denominator_name="Total Newborns (1000-2499g)",
            )
        elif kpi_selection == "Antibiotics for Clinical Sepsis (%)":
            render_antibiotics_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Antibiotics for Clinical Sepsis (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies on Antibiotics (%)":
            render_culture_done_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies on Antibiotics (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies with Clinical Sepsis (%)":
            render_culture_done_sepsis_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies with Clinical Sepsis (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Result Recorded (%)":
            render_culture_result_recorded_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Blood Culture Result Recorded - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia on Admission (%)":
            render_hypothermia_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia on Admission (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia After Admission (%)":
            render_hypothermia_after_admission_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia After Admission (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia Inborn/Outborn":
            render_inborn_outborn_hypothermia_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia Inborn/Outborn - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                comparison_type="facility",
                tei_df=tei_df,
            )
        elif kpi_selection == "Inborn Babies (%)":
            render_inborn_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Inborn Babies (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Newborn Birth Weight Distribution":
            render_newborn_bw_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Newborn Birth Weight Distribution - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
            )
        elif kpi_selection == "Neonatal Mortality Rate (%)":
            render_nmr_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Neonatal Mortality Rate (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )


def render_additional_analytics(
    kpi_selection, filtered_events, facility_uids, bg_color, text_color
):
    """NO ADDITIONAL ANALYTICS"""
    pass


def get_date_range(df, quick_range):
    """Get date range based on quick range selection"""
    from datetime import datetime, timedelta

    today = datetime.now().date()

    if quick_range == "Today":
        return today, today
    elif quick_range == "This Week":
        start_date = today - timedelta(days=today.weekday())
        return start_date, today
    elif quick_range == "Last Week":
        start_date = today - timedelta(days=today.weekday() + 7)
        end_date = start_date + timedelta(days=6)
        return start_date, end_date
    elif quick_range == "This Month":
        start_date = today.replace(day=1)
        return start_date, today
    elif quick_range == "Last Month":
        first_day_current = today.replace(day=1)
        last_day_previous = first_day_current - timedelta(days=1)
        first_day_previous = last_day_previous.replace(day=1)
        return first_day_previous, last_day_previous
    elif quick_range == "This Year":
        start_date = today.replace(month=1, day=1)
        return start_date, today
    elif quick_range == "Last Year":
        start_date = today.replace(year=today.year - 1, month=1, day=1)
        end_date = today.replace(year=today.year - 1, month=12, day=31)
        return start_date, end_date
    else:
        # Custom range - use min/max from dataframe
        if not df.empty:
            # Try to find a date column
            date_series = get_event_date_from_patient_df(df)
            valid_dates = date_series.dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                return min_date, max_date

        # Fallback to last 3 months
        start_date = today - timedelta(days=90)
        return start_date, today


def assign_period(df, date_column, period_label):
    """Assign period based on date column and period label"""
    if df.empty or date_column not in df.columns:
        return df

    df = df.copy()

    if period_label == "Daily":
        df["period"] = df[date_column].dt.strftime("%Y-%m-%d")
        df["period_display"] = df[date_column].dt.strftime("%b %d, %Y")
        df["period_sort"] = df[date_column].dt.strftime("%Y%m%d")
    elif period_label == "Weekly":
        df["period"] = df[date_column].dt.strftime("%Y-W%W")
        df["period_display"] = df[date_column].dt.strftime("Week %W, %Y")
        df["period_sort"] = df[date_column].dt.strftime("%Y%W")
    elif period_label == "Monthly":
        df["period"] = df[date_column].dt.strftime("%Y-%m")
        df["period_display"] = df[date_column].dt.strftime("%b %Y")
        df["period_sort"] = df[date_column].dt.strftime("%Y%m")
    elif period_label == "Quarterly":
        df["period"] = df[date_column].dt.to_period("Q").astype(str)
        df["period_display"] = df[date_column].dt.to_period("Q").astype(str)
        df["period_sort"] = df[date_column].dt.to_period("Q").astype(str)
    elif period_label == "Yearly":
        df["period"] = df[date_column].dt.strftime("%Y")
        df["period_display"] = df[date_column].dt.strftime("%Y")
        df["period_sort"] = df[date_column].dt.strftime("%Y")

    return df


def get_available_aggregations(start_date, end_date):
    """Get available aggregation levels based on date range"""
    days_diff = (end_date - start_date).days

    if days_diff <= 7:
        return ["Daily"]
    elif days_diff <= 31:
        return ["Daily", "Weekly"]
    elif days_diff <= 90:
        return ["Weekly", "Monthly"]
    elif days_diff <= 365:
        return ["Monthly", "Quarterly"]
    else:
        return ["Monthly", "Quarterly", "Yearly"]

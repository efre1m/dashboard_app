# dash_co_newborn.py
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
from newborns_dashboard.kpi_cpap import (
    compute_cpap_kpi,
    render_cpap_trend_chart,
    render_cpap_facility_comparison_chart,
    render_cpap_region_comparison_chart,
)
from newborns_dashboard.kpi_hypothermia import (
    compute_hypothermia_kpi,
    render_hypothermia_trend_chart,
    render_hypothermia_facility_comparison_chart,
    render_hypothermia_region_comparison_chart,
)
from newborns_dashboard.kpi_inborn import (
    compute_inborn_kpi,
    compute_inborn_numerator,  # ✅ IMPORT THE NUMERATOR FUNCTION
    render_inborn_trend_chart,
    render_inborn_facility_comparison_chart,
    render_inborn_region_comparison_chart,
)
from newborns_dashboard.kpi_nmr import (
    compute_nmr_kpi,
    compute_nmr_numerator,  # ✅ IMPORT THE NUMERATOR FUNCTION
    render_nmr_trend_chart,
    render_nmr_facility_comparison_chart,
    render_nmr_region_comparison_chart,
)
from newborns_dashboard.kpi_newborn_bw import (  # ✅ IMPORT NEWBORN BIRTH WEIGHT KPI
    compute_newborn_bw_kpi,
    render_newborn_bw_trend_chart,
    render_newborn_bw_facility_comparison_chart,
    render_newborn_bw_region_comparison_chart,
)

# KPI mapping for KMC coverage, CPAP coverage, Hypothermia, Inborn, NMR, and Birth Weight
KPI_MAPPING = {
    "LBW KMC Coverage (%)": {
        "title": "LBW KMC Coverage (%)",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total LBW Newborns",
    },
    "CPAP Coverage for RDS (%)": {
        "title": "CPAP Coverage for RDS (%)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total RDS Newborns",
    },
    "Hypothermia on Admission (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admissions",
    },
    "Inborn Babies (%)": {
        "title": "Inborn Babies (%)",
        "numerator_name": "Inborn Cases",
        "denominator_name": "Total Admissions",
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

# All KPI options including Birth Weight
KPI_OPTIONS = [
    "LBW KMC Coverage (%)",
    "CPAP Coverage for RDS (%)",
    "Hypothermia on Admission (%)",
    "Inborn Babies (%)",
    "Neonatal Mortality Rate (%)",
    "Newborn Birth Weight Distribution",
]

# KPI Grouping for Tab Navigation - Birth Weight added to "Admission Assessment" group
KPI_GROUPS = {
    "Newborn Care": [
        "LBW KMC Coverage (%)",
        "CPAP Coverage for RDS (%)",
    ],
    "Admission Assessment": [
        "Hypothermia on Admission (%)",
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
    """Render professional tab navigation for Neonatal KPI selection with smaller buttons"""

    # Custom CSS for smaller, more compact button styling
    st.markdown(
        """
    <style>
    /* Make KPI buttons much smaller like normal buttons */
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

    # ✅ FIX: UNIQUE session state key for newborn dashboard only
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
        # Newborn Care KPIs - smaller layout
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "KMC Coverage",
                key="kmc_btn_newborn",  # ✅ Unique key
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "LBW KMC Coverage (%)" else "secondary"
                ),
            ):
                selected_kpi = "LBW KMC Coverage (%)"

        with col2:
            if st.button(
                "CPAP Coverage",
                key="cpap_btn_newborn",  # ✅ Unique key
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "CPAP Coverage for RDS (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "CPAP Coverage for RDS (%)"

    with tab2:
        # Admission Assessment KPIs - three buttons layout
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "Hypothermia",
                key="hypothermia_btn_newborn",  # ✅ Unique key
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
                "Inborn Babies",
                key="inborn_btn_newborn",  # ✅ Unique key
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "Inborn Babies (%)" else "secondary"
                ),
            ):
                selected_kpi = "Inborn Babies (%)"

        with col3:
            if st.button(
                "Birth Weight",
                key="bw_btn_newborn",  # ✅ Unique key
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Newborn Birth Weight Distribution"
                    else "secondary"
                ),
            ):
                selected_kpi = "Newborn Birth Weight Distribution"

    with tab3:
        # Newborn Outcomes KPIs - single button layout
        (col1,) = st.columns(1)

        with col1:
            if st.button(
                "Neonatal Mortality",
                key="nmr_btn_newborn",  # ✅ Unique key
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Neonatal Mortality Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Neonatal Mortality Rate (%)"

    # ✅ FIX: Use the UNIQUE session state key
    if selected_kpi != st.session_state.selected_kpi_NEWBORN_DASHBOARD:
        st.session_state.selected_kpi_NEWBORN_DASHBOARD = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi_NEWBORN_DASHBOARD


def compute_inborn_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of inborn KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "inborn_rate": 0.0,
            "inborn_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    inborn_count = compute_inborn_numerator(df, facility_uids)

    # Count unique TEIs in the filtered dataset
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate inborn rate
    inborn_rate = (
        (inborn_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "inborn_rate": float(inborn_rate),
        "inborn_count": int(inborn_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_nmr_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of NMR KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "nmr_rate": 0.0,
            "dead_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    dead_count = compute_nmr_numerator(df, facility_uids)

    # Count unique TEIs in the filtered dataset
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate NMR rate
    nmr_rate = (
        (dead_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "nmr_rate": float(nmr_rate),
        "dead_count": int(dead_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_bw_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of Birth Weight KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "total_admissions": 0,
            "bw_categories": {},
            "category_rates": {},
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported birth weight function directly
    bw_data = compute_newborn_bw_kpi(df, facility_uids)

    return {
        "total_admissions": bw_data["total_admissions"],
        "bw_categories": bw_data["bw_categories"],
        "category_rates": bw_data["category_rates"],
    }


def render_trend_chart_section(
    kpi_selection,
    filtered_events,
    facility_uids,
    display_names,
    bg_color,
    text_color,
    tei_df=None,
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

    elif kpi_selection == "CPAP Coverage for RDS (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            cpap_data = compute_cpap_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": cpap_data["cpap_rate"],
                    "CPAP Cases": cpap_data["cpap_count"],
                    "Total RDS Newborns": cpap_data["total_rds"],
                }
            )

        group = pd.DataFrame(period_data)
        render_cpap_trend_chart(
            group,
            "period_display",
            "value",
            "CPAP Coverage for RDS (%)",
            bg_color,
            text_color,
            display_names,
            "CPAP Cases",
            "Total RDS Newborns",
            facility_uids,
        )

    elif kpi_selection == "Hypothermia on Admission (%)":
        # ✅ FIX: Pass TEI dataframe to prevent overcounting
        render_hypothermia_trend_chart(
            filtered_events,
            "period_display",
            "Hypothermia on Admission (%)",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,  # ✅ Pass TEI dataframe
        )

    elif kpi_selection == "Inborn Babies (%)":
        # Then render the trend chart
        render_inborn_trend_chart(
            filtered_events,
            "period_display",
            "Inborn Babies Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Neonatal Mortality Rate (%)":
        # Then render the trend chart
        render_nmr_trend_chart(
            filtered_events,
            "period_display",
            "Neonatal Mortality Rate Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Newborn Birth Weight Distribution":
        # ✅ NEW: Render birth weight trend chart
        render_newborn_bw_trend_chart(
            filtered_events,
            "period_display",
            "Newborn Birth Weight Distribution Trend",
            bg_color,
            text_color,
            facility_names=display_names,
            facility_uids=facility_uids,
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
    """Render comparison charts for both national and regional views"""

    if comparison_mode == "facility":
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )
        elif kpi_selection == "CPAP Coverage for RDS (%)":
            render_cpap_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="CPAP Coverage for RDS (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="CPAP Cases",
                denominator_name="Total RDS Newborns",
            )
        elif kpi_selection == "Hypothermia on Admission (%)":
            # ✅ FIX: Pass TEI dataframe to prevent overcounting
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
        elif kpi_selection == "Inborn Babies (%)":
            # Then render the comparison chart
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
        elif kpi_selection == "Neonatal Mortality Rate (%)":
            # Then render the comparison chart
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
        elif kpi_selection == "Newborn Birth Weight Distribution":
            # ✅ NEW: Render birth weight facility comparison chart
            render_newborn_bw_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Newborn Birth Weight Distribution - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
            )

    else:  # region comparison (only for national)
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )
        elif kpi_selection == "CPAP Coverage for RDS (%)":
            render_cpap_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="CPAP Coverage for RDS (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="CPAP Cases",
                denominator_name="Total RDS Newborns",
            )
        elif kpi_selection == "Hypothermia on Admission (%)":
            # ✅ FIX: Pass TEI dataframe to prevent overcounting
            render_hypothermia_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia on Admission (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Inborn Babies (%)":
            # ✅ FIX: Show independent computation first
            st.subheader("📊 Inborn Babies - Region Comparison")

            # Compute overall data for context
            overall_inborn_data = compute_inborn_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Inborn Cases",
                    value=f"{overall_inborn_data['inborn_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admitted Newborns",
                    value=f"{overall_inborn_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Inborn Rate",
                    value=f"{overall_inborn_data['inborn_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_inborn_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Inborn Babies (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Neonatal Mortality Rate (%)":
            # ✅ FIX: Show independent computation first
            st.subheader("📊 Neonatal Mortality Rate - Region Comparison")

            # Compute overall data for context
            overall_nmr_data = compute_nmr_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Dead Cases",
                    value=f"{overall_nmr_data['dead_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admitted Newborns",
                    value=f"{overall_nmr_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall NMR Rate",
                    value=f"{overall_nmr_data['nmr_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_nmr_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Neonatal Mortality Rate (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Newborn Birth Weight Distribution":
            # ✅ NEW: Render birth weight region comparison chart
            st.subheader("📊 Newborn Birth Weight Distribution - Region Comparison")

            # Compute overall data for context
            overall_bw_data = compute_bw_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Total Admissions",
                    value=f"{overall_bw_data['total_admissions']:,}",
                )
            with col2:
                # Show normal birth weight rate (2500-4000g) as key indicator
                normal_bw_rate = overall_bw_data["category_rates"].get("2500_4000", 0)
                st.metric(
                    label="Normal Birth Weight Rate (2500-4000g)",
                    value=f"{normal_bw_rate:.1f}%",
                )

            # Then render the comparison chart
            render_newborn_bw_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Newborn Birth Weight Distribution - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
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

    # ✅ FIX: Use the correct newborn session state key
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi_NEWBORN_DASHBOARD", "LBW KMC Coverage (%)"
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

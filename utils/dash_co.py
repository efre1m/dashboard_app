# utils/dash_co.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations

# REPLACE with this corrected import section:
from utils.kpi_utils import (
    compute_kpis,
    auto_text_color,
    prepare_data_for_trend_chart,
    extract_period_columns,
    get_relevant_date_column_for_kpi,
    get_numerator_denominator_for_kpi,
    DELIVERY_DATE_COL,
    PNC_DATE_COL,
    DISCHARGE_DATE_COL,
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)
from utils.kpi_pph import (
    compute_pph_kpi,
    render_pph_trend_chart,
    render_pph_facility_comparison_chart,
    render_pph_region_comparison_chart,
    render_obstetric_condition_pie_chart,
    compute_pph_numerator,
)
from utils.kpi_uterotonic import (
    compute_uterotonic_kpi,
    render_uterotonic_trend_chart,
    render_uterotonic_facility_comparison_chart,
    render_uterotonic_region_comparison_chart,
    render_uterotonic_type_pie_chart,
    compute_uterotonic_numerator,
    compute_uterotonic_by_type,
)
from utils.kpi_arv import (
    compute_arv_kpi,
    render_arv_trend_chart,
    render_arv_facility_comparison_chart,
    render_arv_region_comparison_chart,
)
from utils.kpi_lbw import (
    compute_lbw_kpi,
    render_lbw_trend_chart,
    render_lbw_facility_comparison_chart,
    render_lbw_region_comparison_chart,
    render_lbw_category_pie_chart,
    LBW_CATEGORIES,
)
from utils.kpi_assisted import (
    compute_assisted_delivery_kpi,
    render_assisted_trend_chart,
    render_assisted_facility_comparison_chart,
    render_assisted_region_comparison_chart,
    compute_assisted_count,
)
from utils.kpi_svd import (
    compute_svd_kpi,
    render_svd_trend_chart,
    render_svd_facility_comparison_chart,
    render_svd_region_comparison_chart,
    compute_svd_count,
)

# ADD THIS IMPORT
from utils.kpi_missing_md import render_missing_md_simple_table

# KPI mapping for comparison charts - used in both files
KPI_MAPPING = {
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
        "title": "IPPCAR (%)",
        "numerator_name": "FP Acceptances",
        "denominator_name": "Total Deliveries",
    },
    "Stillbirth Rate (per 1000 births)": {
        "title": "Stillbirth Rate (per 1000 births)",
        "numerator_name": "Stillbirths",
        "denominator_name": "Total Deliveries",
    },
    "Early Postnatal Care (PNC) Coverage (%)": {
        "title": "Early PNC Coverage (%)",
        "numerator_name": "Early PNC (≤48 hrs)",
        "denominator_name": "Total Deliveries",
    },
    "Institutional Maternal Death Rate (per 100,000 births)": {
        "title": "Maternal Death Rate (per 100,000 births)",
        "numerator_name": "Maternal Deaths",
        "denominator_name": "Total Deliveries",
    },
    "C-Section Rate (%)": {
        "title": "C-Section Rate (%)",
        "numerator_name": "C-Sections",
        "denominator_name": "Total Deliveries",
    },
    "Postpartum Hemorrhage (PPH) Rate (%)": {
        "title": "PPH Rate (%)",
        "numerator_name": "PPH Cases",
        "denominator_name": "Total Deliveries",
    },
    "Delivered women who received uterotonic (%)": {
        "title": "Delivered women who received uterotonic (%)",
        "numerator_name": "Women given uterotonic",
        "denominator_name": "Deliveries",
    },
    "ARV Prophylaxis Rate (%)": {
        "title": "ARV Prophylaxis Rate (%)",
        "numerator_name": "ARV Cases",
        "denominator_name": "HIV-Exposed Infants",
    },
    "Low Birth Weight (LBW) Rate (%)": {
        "title": "Low Birth Weight Rate (%)",
        "numerator_name": "LBW Cases (<2500g)",
        "denominator_name": "Total Weighed Births",
    },
    "Assisted Delivery Rate (%)": {
        "title": "Assisted Delivery Rate (%)",
        "numerator_name": "Assisted Deliveries",
        "denominator_name": "Total Deliveries",
    },
    "Normal Vaginal Delivery (SVD) Rate (%)": {
        "title": "Normal Vaginal Delivery Rate (%)",
        "numerator_name": "SVD Deliveries",
        "denominator_name": "Total Deliveries",
    },
    # ADD THIS NEW KPI
    "Missing Mode of Delivery": {
        "title": "Missing Mode of Delivery",
        "numerator_name": "Missing MD Cases",
        "denominator_name": "Total Deliveries",
    },
}

# KPI Grouping for Tab Navigation
KPI_GROUPS = {
    "Mortality": [
        "Institutional Maternal Death Rate (per 100,000 births)",
        "Stillbirth Rate (per 1000 births)",
    ],
    "Complications": [
        "Postpartum Hemorrhage (PPH) Rate (%)",
        "Low Birth Weight (LBW) Rate (%)",
    ],
    "Care": [
        "C-Section Rate (%)",
        "Delivered women who received uterotonic (%)",
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
        "Early Postnatal Care (PNC) Coverage (%)",
        "ARV Prophylaxis Rate (%)",
        "Assisted Delivery Rate (%)",
        "Normal Vaginal Delivery (SVD) Rate (%)",
    ],
    # ADD THIS NEW GROUP
    "Missing": [
        "Missing Mode of Delivery",
    ],
}


def get_text_color(bg_color):
    """Get auto text color for background - used in both files"""
    return auto_text_color(bg_color)


def get_kpi_config(kpi_selection):
    """Get KPI configuration - used in both files"""
    return KPI_MAPPING.get(kpi_selection, {})


# Common KPI options used in both files
KPI_OPTIONS = [
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "Stillbirth Rate (per 1000 births)",
    "Early Postnatal Care (PNC) Coverage (%)",
    "Institutional Maternal Death Rate (per 100,000 births)",
    "C-Section Rate (%)",
    "Postpartum Hemorrhage (PPH) Rate (%)",
    "Delivered women who received uterotonic (%)",
    "ARV Prophylaxis Rate (%)",
    "Low Birth Weight (LBW) Rate (%)",
    "Assisted Delivery Rate (%)",
    "Normal Vaginal Delivery (SVD) Rate (%)",
    "Missing Mode of Delivery",  # ADD THIS
]


def render_kpi_tab_navigation():
    """Render professional tab navigation for KPI selection without duplication"""

    # Custom CSS for professional tab styling and active button highlighting
    # Using !important to override national.css styles
    st.markdown(
        """
    <style>
    /* Custom button styling for active state - using !important to override */
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
    
    /* Ensure our styles take precedence */
    .stButton button {
        transition: all 0.3s ease !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # KPI Grouping for Tab Navigation
    KPI_GROUPS = {
        "📉 Mortality": [
            "Institutional Maternal Death Rate (per 100,000 births)",
            "Stillbirth Rate (per 1000 births)",
        ],
        "🚨 Complications": [
            "Postpartum Hemorrhage (PPH) Rate (%)",
            "Low Birth Weight (LBW) Rate (%)",
        ],
        "🏥 Care": [
            "C-Section Rate (%)",
            "Delivered women who received uterotonic (%)",
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "Early Postnatal Care (PNC) Coverage (%)",
            "ARV Prophylaxis Rate (%)",
            "Assisted Delivery Rate (%)",
            "Normal Vaginal Delivery (SVD) Rate (%)",
        ],
        # ADD THIS NEW GROUP FOR MISSING DATA
        "❓ Missing": [
            "Missing Mode of Delivery",
        ],
    }

    # Initialize session state for KPI selection
    if "selected_kpi" not in st.session_state:
        st.session_state.selected_kpi = (
            "Institutional Maternal Death Rate (per 100,000 births)"
        )

    # Create main KPI group tabs - ADD 4TH TAB
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 **Mortality**", "🚨 **Complications**", "🏥 **Care**", "❓ **Missing**"]
    )

    selected_kpi = st.session_state.selected_kpi

    with tab1:
        # Mortality KPIs - Direct selection without instructional text
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 Maternal Death Rate",
                key="maternal_death_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi
                    == "Institutional Maternal Death Rate (per 100,000 births)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Institutional Maternal Death Rate (per 100,000 births)"

        with col2:
            if st.button(
                "📊 Stillbirth Rate",
                key="stillbirth_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Stillbirth Rate (per 1000 births)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Stillbirth Rate (per 1000 births)"

    with tab2:
        # Complications KPIs - Direct selection without instructional text
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 Postpartum Hemorrhage",
                key="pph_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Postpartum Hemorrhage (PPH) Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Postpartum Hemorrhage (PPH) Rate (%)"

        with col2:
            if st.button(
                "📊 Low Birth Weight",
                key="lbw_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Low Birth Weight (LBW) Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Low Birth Weight (LBW) Rate (%)"

    with tab3:
        # Care KPIs - Direct selection without instructional text
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 C-Section Rate",
                key="csection_btn",
                use_container_width=True,
                type="primary" if selected_kpi == "C-Section Rate (%)" else "secondary",
            ):
                selected_kpi = "C-Section Rate (%)"

            if st.button(
                "📊 Uterotonic Administration",
                key="uterotonic_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Delivered women who received uterotonic (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Delivered women who received uterotonic (%)"

            if st.button(
                "📊 Contraceptive Acceptance",
                key="contraceptive_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi
                    == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"
                    else "secondary"
                ),
            ):
                selected_kpi = (
                    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"
                )

        with col2:
            if st.button(
                "📊 PNC Coverage",
                key="pnc_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Early Postnatal Care (PNC) Coverage (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Early Postnatal Care (PNC) Coverage (%)"

            if st.button(
                "📊 ARV Prophylaxis",
                key="arv_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "ARV Prophylaxis Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "ARV Prophylaxis Rate (%)"

            if st.button(
                "📊 Assisted Delivery",
                key="assisted_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Assisted Delivery Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Assisted Delivery Rate (%)"

            if st.button(
                "📊 Normal Vaginal Delivery",
                key="svd_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Normal Vaginal Delivery (SVD) Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Normal Vaginal Delivery (SVD) Rate (%)"

    with tab4:
        # Missing Data KPIs - Just show the Missing Mode of Delivery button
        st.markdown("### Missing Data Analysis")
        st.markdown("Analyze data completeness and missing information")

        if st.button(
            "📊 Missing Mode of Delivery",
            key="missing_md_btn",
            use_container_width=True,
            type=(
                "primary" if selected_kpi == "Missing Mode of Delivery" else "secondary"
            ),
        ):
            selected_kpi = "Missing Mode of Delivery"

    # Update session state with final selection
    if selected_kpi != st.session_state.selected_kpi:
        st.session_state.selected_kpi = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi


def get_period_columns(df):
    """Get period column names from transformed data"""
    period_col = None
    period_display_col = None
    period_sort_col = None

    # Look for period columns in the transformed data
    for col in df.columns:
        if "period_display" in col:
            period_display_col = col
        elif "period_sort" in col:
            period_sort_col = col
        elif col == "period":
            period_col = col

    # Fallback to first column if needed
    if not period_col:
        period_col = "period" if "period" in df.columns else df.columns[0]

    return period_col, period_display_col, period_sort_col


def render_trend_chart_section(
    kpi_selection,
    patient_df,
    facility_uids,
    display_names,
    bg_color,
    text_color,
    comparison_mode="overall",
    facilities_by_region=None,
    region_names=None,
    show_table=False,
):
    """Render the trend chart based on KPI selection - SIMPLIFIED FIX"""

    # SPECIAL HANDLING FOR MISSING MODE OF DELIVERY
    if kpi_selection == "Missing Mode of Delivery":
        render_missing_md_simple_table(
            df=patient_df,
            facility_uids=facility_uids,
            display_names=display_names,
            comparison_mode=comparison_mode,
            facilities_by_region=facilities_by_region,
            region_names=region_names,
        )
        return

    # Get KPI configuration for labels
    kpi_config = get_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)

    # Prepare data for trend chart
    prepared_df = prepare_data_for_trend_chart(patient_df, kpi_selection, facility_uids)

    if prepared_df.empty:
        st.info("⚠️ No data available for trend analysis.")
        return

    # Ensure period columns exist
    if "period_display" not in prepared_df.columns:
        prepared_df["period_display"] = prepared_df["event_date"].dt.strftime("%b-%y")
    if "period_sort" not in prepared_df.columns:
        prepared_df["period_sort"] = prepared_df["event_date"].dt.strftime("%Y%m")

    # Get unique periods in order
    unique_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Create period data
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period
        period_df = prepared_df[prepared_df["period_display"] == period_display]

        if not period_df.empty:
            # Get numerator and denominator
            numerator, denominator, _ = get_numerator_denominator_for_kpi(
                period_df, kpi_selection, facility_uids
            )

            # Calculate value based on KPI type
            if "IPPCAR" in kpi_selection or "%" in kpi_selection:
                value = (numerator / denominator * 100) if denominator > 0 else 0
            elif "Stillbirth Rate" in kpi_selection:
                value = (numerator / denominator * 1000) if denominator > 0 else 0
            elif "Maternal Death Rate" in kpi_selection:
                value = (numerator / denominator * 100000) if denominator > 0 else 0
            else:
                value = 0

            period_data.append(
                {
                    "period": period_display,
                    "period_display": period_display,
                    "period_sort": period_sort,
                    "value": value,
                    "numerator": int(numerator),  # Ensure integer
                    "denominator": int(denominator),  # Ensure integer
                }
            )

    if not period_data:
        st.info("⚠️ No period data available for chart.")
        return

    # Create DataFrame
    group = pd.DataFrame(period_data)

    # Sort by period_sort
    group = group.sort_values("period_sort")

    # Render the chart
    try:
        # Call the appropriate chart function based on KPI
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Assisted Delivery Rate (%)":
            render_assisted_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
            render_svd_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
        else:
            # For all other KPIs, use the standard trend chart
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
                show_table=show_table,
            )
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")
        import traceback

        st.error(traceback.format_exc())


def _create_period_row(
    kpi_selection, period, period_display, kpi_data, numerator_label, denominator_label
):
    """Create a standardized period row based on KPI type"""
    if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["ippcar"],
            "numerator": kpi_data["fp_acceptance"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Stillbirth Rate (per 1000 births)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["stillbirth_rate"],
            "numerator": kpi_data["stillbirths"],
            "denominator": kpi_data["total_deliveries_sb"],
        }
    elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["pnc_coverage"],
            "numerator": kpi_data["early_pnc"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["maternal_death_rate"],
            "numerator": kpi_data["maternal_deaths"],
            "denominator": kpi_data["total_deliveries_md"],
        }
    elif kpi_selection == "C-Section Rate (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["csection_rate"],
            "numerator": kpi_data["csection_deliveries"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["pph_rate"],
            "numerator": kpi_data["pph_count"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Delivered women who received uterotonic (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["uterotonic_rate"],
            "numerator": kpi_data["uterotonic_count"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Assisted Delivery Rate (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["assisted_delivery_rate"],
            "numerator": kpi_data["assisted_deliveries"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
        return {
            "period": period,
            "period_display": period_display,
            "value": kpi_data["svd_rate"],
            "numerator": kpi_data["svd_deliveries"],
            "denominator": kpi_data["total_deliveries"],
        }
    elif kpi_selection == "Missing Mode of Delivery":
        return {}
    return {}


def _render_kpi_chart_with_labels(
    kpi_selection,
    group,
    bg_color,
    text_color,
    display_names,
    facility_uids,
    chart_title,
    numerator_label,
    denominator_label,
    show_table=False,
):
    """Render the appropriate chart based on KPI selection with proper labels"""
    try:
        if (
            kpi_selection
            == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"
        ):
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Stillbirth Rate (per 1000 births)":
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "C-Section Rate (%)":
            render_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Assisted Delivery Rate (%)":
            render_assisted_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
            render_svd_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,  # PASS THE ACTUAL LABEL
                denominator_label,  # PASS THE ACTUAL LABEL
                facility_uids,
                show_table=show_table,
            )
        # Note: Missing Mode of Delivery is handled separately
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")


def _get_default_kpi_data(kpi_selection):
    """Get default empty data for different KPI types"""
    defaults = {
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
            "ippcar": 0,
            "fp_acceptance": 0,
            "total_deliveries": 0,
        },
        "Stillbirth Rate (per 1000 births)": {
            "stillbirth_rate": 0,
            "stillbirths": 0,
            "total_deliveries_sb": 0,
        },
        "Early Postnatal Care (PNC) Coverage (%)": {
            "pnc_coverage": 0,
            "early_pnc": 0,
            "total_deliveries": 0,
        },
        "Institutional Maternal Death Rate (per 100,000 births)": {
            "maternal_death_rate": 0,
            "maternal_deaths": 0,
            "total_deliveries_md": 0,
        },
        "C-Section Rate (%)": {
            "csection_rate": 0,
            "csection_deliveries": 0,
            "total_deliveries": 0,
        },
        "Postpartum Hemorrhage (PPH) Rate (%)": {
            "pph_rate": 0.0,
            "pph_count": 0,
            "total_deliveries": 0,
        },
        "Delivered women who received uterotonic (%)": {
            "uterotonic_rate": 0.0,
            "uterotonic_count": 0,
            "total_deliveries": 0,
            "uterotonic_types": {
                "Ergometrine": 0,
                "Oxytocin": 0,
                "Misoprostol": 0,
                "total": 0,
            },
        },
        "Assisted Delivery Rate (%)": {
            "assisted_delivery_rate": 0.0,
            "assisted_deliveries": 0,
            "total_deliveries": 0,
        },
        "Normal Vaginal Delivery (SVD) Rate (%)": {
            "svd_rate": 0.0,
            "svd_deliveries": 0,
            "total_deliveries": 0,
        },
        "Missing Mode of Delivery": {
            "missing_md_rate": 0.0,
            "missing_md_count": 0,
            "total_deliveries": 0,
        },
    }
    return defaults.get(kpi_selection, {})


def render_comparison_chart(
    kpi_selection,
    patient_df=None,
    comparison_mode="facility",
    display_names=None,
    facility_uids=None,
    facilities_by_region=None,
    bg_color="#FFFFFF",
    text_color=None,
    is_national=False,
    filtered_events=None,
    show_table=False,
):
    """Render comparison charts for both national and regional views - UPDATED for transformed data"""

    # USE filtered_events IF PROVIDED, OTHERWISE USE patient_df
    df_to_use = filtered_events if filtered_events is not None else patient_df

    if df_to_use is None or df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    if kpi_selection == "Missing Mode of Delivery":
        return

    kpi_config = get_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)

    # Check if orgUnit column exists in the data
    if "orgUnit" not in df_to_use.columns:
        st.error(
            "❌ Column 'orgUnit' not found in the data. Cannot perform comparison."
        )
        st.info(
            "Please ensure your data contains an 'orgUnit' column with facility IDs."
        )
        return

    # Prepare data for the chart
    prepared_df = prepare_data_for_trend_chart(df_to_use, kpi_selection, facility_uids)

    if prepared_df.empty:
        st.info("⚠️ No prepared data available for comparison.")
        return

    # Create comparison data
    comparison_data = []

    # Get all unique periods
    if "period_display" not in prepared_df.columns:
        prepared_df["period_display"] = prepared_df["event_date"].dt.strftime("%b-%y")

    all_periods = prepared_df["period_display"].unique()

    if comparison_mode == "facility":
        # Facility comparison
        for period in all_periods:
            period_df = prepared_df[prepared_df["period_display"] == period]

            for facility_uid, facility_name in zip(facility_uids, display_names):
                facility_df = period_df[period_df["orgUnit"] == facility_uid]
                if not facility_df.empty:
                    numerator, denominator, value = get_numerator_denominator_for_kpi(
                        facility_df, kpi_selection, [facility_uid]
                    )

                    # Calculate value based on KPI type
                    if "IPPCAR" in kpi_selection:
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )
                    elif "Stillbirth Rate" in kpi_selection:
                        value = (
                            (numerator / denominator * 1000) if denominator > 0 else 0
                        )
                    elif "Maternal Death Rate" in kpi_selection:
                        value = (
                            (numerator / denominator * 100000) if denominator > 0 else 0
                        )
                    elif "%" in kpi_selection:
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )
                    else:
                        value = 0

                    comparison_data.append(
                        {
                            "period_display": period,
                            "Facility": facility_name,
                            "value": value,
                            "numerator": numerator,
                            "denominator": denominator,
                        }
                    )

        if not comparison_data:
            st.info("⚠️ No comparison data available.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # Call the appropriate chart function
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        elif kpi_selection == "Assisted Delivery Rate (%)":
            render_assisted_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
            render_svd_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
        else:
            render_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
                show_table=show_table,
            )
    else:
        # Region comparison (only for national)
        st.info("Region comparison is only available in national view")


def render_additional_analytics(
    kpi_selection, patient_df, facility_uids, bg_color, text_color
):
    """Render additional analytics charts - UPDATED for transformed data"""
    if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        render_obstetric_condition_pie_chart(
            patient_df, facility_uids, bg_color, text_color
        )
    elif kpi_selection == "Delivered women who received uterotonic (%)":
        render_uterotonic_type_pie_chart(
            patient_df, facility_uids, bg_color, text_color
        )
    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        render_lbw_category_pie_chart(patient_df, facility_uids, bg_color, text_color)


def normalize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single datetime column 'event_date' exists for transformed data"""
    if df.empty:
        return df

    df = df.copy()

    # For transformed data, look for date columns - UPDATED COLUMN NAMES
    date_cols = [
        col
        for col in df.columns
        if "event_date" in col.lower() or "eventdate" in col.lower()
    ]

    for col in date_cols:
        try:
            # Try parsing ISO format
            df["event_date"] = pd.to_datetime(df[col], errors="coerce")
            # If that fails, try US format
            if df["event_date"].isna().all():
                df["event_date"] = pd.to_datetime(
                    df[col], format="%m/%d/%Y", errors="coerce"
                )
            if not df["event_date"].isna().all():
                break
        except:
            continue

    # If no date column found, use current date
    if "event_date" not in df.columns or df["event_date"].isna().all():
        df["event_date"] = pd.Timestamp.now().normalize()

    return df


def normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure enrollmentDate is datetime from various formats."""
    if df.empty or "enrollment_date" not in df.columns:  # UPDATED: enrollment_date
        return df
    df = df.copy()
    df["enrollment_date"] = pd.to_datetime(
        df["enrollment_date"], errors="coerce"
    )  # UPDATED
    return df


# ========== UPDATED FILTER CONTROLS FOR TRANSFORMED DATA ==========


def render_simple_filter_controls(patient_df, container=None, context="default"):
    """Simple filter controls without KPI selection"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
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
        _df_for_dates = (
            patient_df[["event_date"]].copy()
            if not patient_df.empty and "event_date" in patient_df.columns
            else pd.DataFrame()
        )
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level - CRITICAL SECTION
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )
    default_index = (
        available_aggregations.index("Monthly")
        if "Monthly" in available_aggregations
        else 0
    )

    # Get current value from session state if exists
    current_period_label = st.session_state.get("period_label", "Monthly")
    if current_period_label in available_aggregations:
        default_index = available_aggregations.index(current_period_label)

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",
    )

    # ✅ CRITICAL FIX: Save period_label to session state
    st.session_state.period_label = filters["period_label"]
    print(
        f"🔧 DEBUG dash_co: Set period_label to session_state: {filters['period_label']}"
    )

    # Also save to filters dictionary in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters["period_label"] = filters["period_label"]
    st.session_state.filters["start_date"] = filters["start_date"]
    st.session_state.filters["end_date"] = filters["end_date"]

    print(f"🔧 DEBUG dash_co: Available aggregations: {available_aggregations}")
    print(f"🔧 DEBUG dash_co: Selected period_label: {filters['period_label']}")

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add placeholder for kpi_selection
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi", "Institutional Maternal Death Rate (per 100,000 births)"
    )

    # Add show_table option
    filters["show_table"] = container.checkbox(
        "📊 Show Data Table",
        value=False,
        key=f"show_table{key_suffix}",
        help="Display the data table below the chart",
    )

    return filters


def _get_simple_date_range(patient_df):
    """Get min/max dates from transformed dataframe"""
    import datetime

    if not patient_df.empty:
        # Try multiple date columns
        date_columns = []
        for col in patient_df.columns:
            if "event_date" in col.lower():
                date_columns.append(col)
            elif "date" in col.lower() and "update" not in col.lower():
                date_columns.append(col)

        print(f"🔧 DEBUG _get_simple_date_range: Found date columns: {date_columns}")

        for date_col in date_columns:
            if date_col in patient_df.columns:
                # Convert to datetime
                dates = pd.to_datetime(patient_df[date_col], errors="coerce")
                valid_dates = dates.dropna()

                if not valid_dates.empty:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()

                    if hasattr(min_date, "date"):
                        min_date = min_date.date()
                    if hasattr(max_date, "date"):
                        max_date = max_date.date()

                    print(f"🔧 DEBUG _get_simple_date_range: Using {date_col}")
                    print(f"   Min date: {min_date}, Max date: {max_date}")
                    return min_date, max_date

    # Fallback to current date
    today = datetime.date.today()
    print(f"⚠️ _get_simple_date_range: Using fallback dates: {today} to {today}")
    return today, today


def apply_simple_filters(patient_df, filters, facility_uids=None):
    """Apply simple filters to transformed dataframe"""
    if patient_df.empty:
        print("⚠️ apply_simple_filters: Patient dataframe is empty")
        return patient_df

    df = patient_df.copy()

    # Apply date filters
    start_datetime = pd.to_datetime(filters["start_date"])
    end_datetime = pd.to_datetime(filters["end_date"])

    print(f"🔧 DEBUG apply_simple_filters:")
    print(f"   Date range: {start_datetime} to {end_datetime}")
    print(f"   Original rows: {len(df)}")

    # Check if event_date column exists
    if "event_date" not in df.columns:
        print(f"❌ apply_simple_filters: 'event_date' column not found!")
        print(f"   Available columns: {list(df.columns)}")
        # Try to find date columns
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            print(f"   Found date columns: {date_cols}")
            # Use first date column
            df["event_date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            print(f"❌ No date columns found at all!")
            return df

    # Filter by date range
    initial_count = len(df)
    df = df[
        (df["event_date"] >= start_datetime) & (df["event_date"] <= end_datetime)
    ].copy()
    filtered_count = len(df)

    print(
        f"   After date filtering: {filtered_count} rows (removed {initial_count - filtered_count})"
    )

    # Apply facility filter if provided
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        if "orgUnit" in df.columns:
            facility_count = len(df)
            df = df[df["orgUnit"].isin(facility_uids)]
            print(
                f"   After facility filtering: {len(df)} rows (removed {facility_count - len(df)})"
            )
        else:
            print(f"⚠️ 'orgUnit' column not found, cannot filter by facility")

    # ✅ CRITICAL: Ensure period_label is in filters
    if "period_label" not in filters:
        print(f"❌ apply_simple_filters: period_label not in filters!")
        print(f"   Available keys in filters: {list(filters.keys())}")
        # Try to get from session state
        if "period_label" in st.session_state:
            filters["period_label"] = st.session_state.period_label
            print(f"   Got period_label from session_state: {filters['period_label']}")
        else:
            filters["period_label"] = "Monthly"
            print(f"⚠️ Using default period_label: Monthly")

    print(
        f"🔧 DEBUG apply_simple_filters: Using period_label = '{filters['period_label']}'"
    )

    # ✅ CRITICAL: Save to session state for prepare_data_for_trend_chart
    st.session_state.period_label = filters["period_label"]
    print(
        f"🔧 DEBUG apply_simple_filters: Saved to session_state: '{filters['period_label']}'"
    )

    # Also ensure filters dict is in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters.update(filters)

    # ✅ DEBUG: Check event_date column before assign_period
    if "event_date" in df.columns:
        print(f"🔧 DEBUG apply_simple_filters: 'event_date' sample dates:")
        print(f"   First 5: {df['event_date'].head(5).tolist()}")
        print(f"   Data type: {df['event_date'].dtype}")
        print(f"   Not null: {df['event_date'].notna().sum()}/{len(df)}")
    else:
        print(f"❌ ERROR: 'event_date' column missing before assign_period!")

    # Assign period - THIS IS WHERE THE ACTUAL GROUPING HAPPENS
    print(f"🔧 DEBUG apply_simple_filters: Calling assign_period with:")
    print(f"   df shape: {df.shape}")
    print(f"   date_col: 'event_date'")
    print(f"   period_label: '{filters['period_label']}'")

    try:
        df = assign_period(df, "event_date", filters["period_label"])

        # ✅ DEBUG: Check what periods were created
        if "period_display" in df.columns:
            unique_periods = df["period_display"].unique()
            print(f"✅ assign_period created {len(unique_periods)} unique periods:")
            for period in sorted(unique_periods):
                count = df[df["period_display"] == period].shape[0]
                print(f"   {period}: {count} rows")
        else:
            print(f"❌ assign_period did not create 'period_display' column!")
            print(f"   Columns after assign_period: {list(df.columns)}")

    except Exception as e:
        print(f"❌ ERROR in assign_period: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")

    print(f"✅ apply_simple_filters complete. Final shape: {df.shape}")
    return df


# ========== NEW FUNCTIONS FOR TRANSFORMED DATA ==========


def get_period_data_for_kpi(kpi_selection, patient_df, facility_uids):
    """Get period-based data for a specific KPI from transformed data"""
    # Extract relevant date column
    date_column = get_relevant_date_column_for_kpi(kpi_selection)

    # Extract period columns
    df_with_periods = extract_period_columns(patient_df, date_column)

    if df_with_periods.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df_with_periods = df_with_periods[
            df_with_periods["orgUnit"].isin(facility_uids)
        ]

    # Remove rows without valid dates
    df_with_periods = df_with_periods[df_with_periods["event_date"].notna()]

    # Sort by period
    df_with_periods = df_with_periods.sort_values("period_sort")

    return df_with_periods


def compute_kpi_for_period(kpi_selection, period_df, facility_uids):
    """Compute KPI for a specific period using transformed data"""
    if period_df.empty:
        return _get_default_kpi_data(kpi_selection)

    # Use the get_numerator_denominator_for_kpi function
    numerator, denominator, value = get_numerator_denominator_for_kpi(
        period_df, kpi_selection, facility_uids
    )

    # Create appropriate KPI data structure
    if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
        return {
            "ippcar": value,
            "fp_acceptance": numerator,
            "total_deliveries": denominator,
        }
    elif kpi_selection == "Stillbirth Rate (per 1000 births)":
        return {
            "stillbirth_rate": value,
            "stillbirths": numerator,
            "total_deliveries_sb": denominator,
        }
    elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
        return {
            "pnc_coverage": value,
            "early_pnc": numerator,
            "total_deliveries": denominator,
        }
    elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
        return {
            "maternal_death_rate": value,
            "maternal_deaths": numerator,
            "total_deliveries_md": denominator,
        }
    elif kpi_selection == "C-Section Rate (%)":
        return {
            "csection_rate": value,
            "csection_deliveries": numerator,
            "total_deliveries": denominator,
        }
    elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        return compute_pph_kpi(period_df, facility_uids)
    elif kpi_selection == "Delivered women who received uterotonic (%)":
        return compute_uterotonic_kpi(period_df, facility_uids)
    elif kpi_selection == "ARV Prophylaxis Rate (%)":
        return compute_arv_kpi(period_df, facility_uids)
    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        return compute_lbw_kpi(period_df, facility_uids)
    elif kpi_selection == "Assisted Delivery Rate (%)":
        return compute_assisted_delivery_kpi(period_df, facility_uids)
    elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
        return compute_svd_kpi(period_df, facility_uids)
    else:
        return _get_default_kpi_data(kpi_selection)


def get_date_column_from_patient_df(df, kpi_selection):
    """Get the appropriate date column from patient-level data based on KPI"""
    # Map KPIs to their relevant date columns - UPDATED WITH NEW COLUMN NAMES
    if kpi_selection in [
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
        "Early Postnatal Care (PNC) Coverage (%)",
    ]:
        # PNC-related KPIs use PNC date
        for col in [PNC_DATE_COL, "event_date_postpartum_care"]:  # UPDATED
            if col in df.columns:
                return col

    elif kpi_selection in [
        "C-Section Rate (%)",
        "Postpartum Hemorrhage (PPH) Rate (%)",
        "Delivered women who received uterotonic (%)",
        "Assisted Delivery Rate (%)",
        "Normal Vaginal Delivery (SVD) Rate (%)",
        "Stillbirth Rate (per 1000 births)",
        "Institutional Maternal Death Rate (per 100,000 births)",
    ]:
        # Delivery-related KPIs use delivery date
        for col in [DELIVERY_DATE_COL, "event_date_delivery_summary"]:  # UPDATED
            if col in df.columns:
                return col

    elif kpi_selection == "ARV Prophylaxis Rate (%)":
        # ARV-related KPIs might use various dates
        for col in [
            PNC_DATE_COL,
            DELIVERY_DATE_COL,
            "event_date_postpartum_care",  # UPDATED
            "event_date_delivery_summary",  # UPDATED
        ]:
            if col in df.columns:
                return col

    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        # LBW uses birth weight from delivery
        for col in [DELIVERY_DATE_COL, "event_date_delivery_summary"]:  # UPDATED
            if col in df.columns:
                return col

    # Fallback to any event_date column
    for col in df.columns:
        if "event_date" in col.lower():
            return col

    # Last resort
    return "event_date"

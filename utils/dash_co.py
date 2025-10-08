# common_dash.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import compute_kpis, auto_text_color
from utils.kpi_pph import (
    compute_pph_kpi,
    render_pph_trend_chart,
    render_pph_facility_comparison_chart,
    render_pph_region_comparison_chart,
    render_obstetric_condition_pie_chart,
)
from utils.kpi_uterotonic import (
    compute_uterotonic_kpi,
    render_uterotonic_trend_chart,
    render_uterotonic_facility_comparison_chart,
    render_uterotonic_region_comparison_chart,
    render_uterotonic_type_pie_chart,
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
from utils.kpi_utils import (
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)

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
        "denominator_name": "Total Births",
    },
    "Early Postnatal Care (PNC) Coverage (%)": {
        "title": "Early PNC Coverage (%)",
        "numerator_name": "Early PNC (≤48 hrs)",
        "denominator_name": "Total Deliveries",
    },
    "Institutional Maternal Death Rate (per 100,000 births)": {
        "title": "Maternal Death Rate (per 100,000 births)",
        "numerator_name": "Maternal Deaths",
        "denominator_name": "Live Births",
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
        ],
    }

    # Initialize session state for KPI selection
    if "selected_kpi" not in st.session_state:
        st.session_state.selected_kpi = (
            "Institutional Maternal Death Rate (per 100,000 births)"
        )

    # Create main KPI group tabs
    tab1, tab2, tab3 = st.tabs(
        ["📉 **Mortality**", "🚨 **Complications**", "🏥 **Care**"]
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

    # Update session state with final selection
    if selected_kpi != st.session_state.selected_kpi:
        st.session_state.selected_kpi = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi


def render_trend_chart_section(
    kpi_selection, filtered_events, facility_uids, display_names, bg_color, text_color
):
    """Render the trend chart based on KPI selection - same logic in both files"""

    if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_kpis(x, facility_uids)["ippcar"],
                        "FP Acceptances": compute_kpis(x, facility_uids)[
                            "fp_acceptance"
                        ],
                        "Total Deliveries": compute_kpis(x, facility_uids)[
                            "total_deliveries"
                        ],
                    }
                )
            )
            .reset_index(drop=True)
        )
        render_trend_chart(
            group,
            "period_display",
            "value",
            "IPPCAR (%)",
            bg_color,
            text_color,
            display_names,
            "FP Acceptances",
            "Total Deliveries",
            facility_uids,
        )

    elif kpi_selection == "Stillbirth Rate (per 1000 births)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_kpis(x, facility_uids)["stillbirth_rate"],
                        "Stillbirths": compute_kpis(x, facility_uids)["stillbirths"],
                        "Total Births": compute_kpis(x, facility_uids)["total_births"],
                    }
                )
            )
            .reset_index(drop=True)
        )
        render_trend_chart(
            group,
            "period_display",
            "value",
            "Stillbirth Rate (per 1000 births)",
            bg_color,
            text_color,
            display_names,
            "Stillbirths",
            "Total Births",
            facility_uids,
        )

    elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_kpis(x, facility_uids)["pnc_coverage"],
                        "Early PNC (≤48 hrs)": compute_kpis(x, facility_uids)[
                            "early_pnc"
                        ],
                        "Total Deliveries": compute_kpis(x, facility_uids)[
                            "total_deliveries_pnc"
                        ],
                    }
                )
            )
            .reset_index(drop=True)
        )
        render_trend_chart(
            group,
            "period_display",
            "value",
            "Early PNC Coverage (%)",
            bg_color,
            text_color,
            display_names,
            "Early PNC (≤48 hrs)",
            "Total Deliveries",
            facility_uids,
        )

    elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_kpis(x, facility_uids)["maternal_death_rate"],
                        "Maternal Deaths": compute_kpis(x, facility_uids)[
                            "maternal_deaths"
                        ],
                        "Live Births": compute_kpis(x, facility_uids)["live_births"],
                    }
                )
            )
            .reset_index(drop=True)
        )
        render_trend_chart(
            group,
            "period_display",
            "value",
            "Maternal Death Rate (per 100,000 births)",
            bg_color,
            text_color,
            display_names,
            "Maternal Deaths",
            "Live Births",
            facility_uids,
        )

    elif kpi_selection == "C-Section Rate (%)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_kpis(x, facility_uids)["csection_rate"],
                        "C-Sections": compute_kpis(x, facility_uids)[
                            "csection_deliveries"
                        ],
                        "Total Deliveries": compute_kpis(x, facility_uids)[
                            "total_deliveries_cs"
                        ],  # FIXED
                    }
                )
            )
            .reset_index(drop=True)
        )

        if group.empty:
            group = pd.DataFrame(
                {
                    "period": [],
                    "period_display": [],
                    "value": [],
                    "C-Sections": [],
                    "Total Deliveries": [],  # FIXED
                }
            )

        render_trend_chart(
            group,
            "period_display",
            "value",
            "C-Section Rate (%)",
            bg_color,
            text_color,
            display_names,
            "C-Sections",
            "Total Deliveries",
            facility_uids,
        )

    elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        group = (
            filtered_events.groupby(["period", "period_display"], as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "value": compute_pph_kpi(x, facility_uids)["pph_rate"],
                        "PPH Cases": compute_pph_kpi(x, facility_uids)["pph_count"],
                        "Total Deliveries": compute_pph_kpi(x, facility_uids)[
                            "total_deliveries"
                        ],
                    }
                )
            )
            .reset_index(drop=True)
        )
        render_pph_trend_chart(
            group,
            "period_display",
            "value",
            "PPH Rate (%)",
            bg_color,
            text_color,
            display_names,
            "PPH Cases",
            "Total Deliveries",
            facility_uids,
        )

    elif kpi_selection == "Delivered women who received uterotonic (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            kpi_data = compute_uterotonic_kpi(period_df, facility_uids)
            total_deliveries = kpi_data["total_deliveries"]

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": kpi_data["uterotonic_rate"],
                    "Uterotonic Cases": kpi_data["uterotonic_count"],
                    "Total Deliveries": total_deliveries,
                    "ergometrine_rate": (
                        (
                            kpi_data["uterotonic_types"]["Ergometrine"]
                            / total_deliveries
                            * 100
                        )
                        if total_deliveries > 0
                        else 0
                    ),
                    "oxytocin_rate": (
                        (
                            kpi_data["uterotonic_types"]["Oxytocin"]
                            / total_deliveries
                            * 100
                        )
                        if total_deliveries > 0
                        else 0
                    ),
                    "misoprostol_rate": (
                        (
                            kpi_data["uterotonic_types"]["Misoprostol"]
                            / total_deliveries
                            * 100
                        )
                        if total_deliveries > 0
                        else 0
                    ),
                    "ergometrine_count": kpi_data["uterotonic_types"]["Ergometrine"],
                    "oxytocin_count": kpi_data["uterotonic_types"]["Oxytocin"],
                    "misoprostol_count": kpi_data["uterotonic_types"]["Misoprostol"],
                }
            )

        group = pd.DataFrame(period_data)
        render_uterotonic_trend_chart(
            group,
            "period_display",
            "value",
            "Uterotonic Administration Rate (%)",
            bg_color,
            text_color,
            display_names,
            "Uterotonic Cases",
            "Total Deliveries",
            facility_uids,
        )

    elif kpi_selection == "ARV Prophylaxis Rate (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            arv_data = compute_arv_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": arv_data["arv_rate"],
                    "ARV Cases": arv_data["arv_count"],
                    "HIV-Exposed Infants": arv_data["hiv_exposed_infants"],
                }
            )

        group = pd.DataFrame(period_data)
        render_arv_trend_chart(
            group,
            "period_display",
            "value",
            "ARV Prophylaxis Rate (%)",
            bg_color,
            text_color,
            display_names,
            "ARV Cases",
            "HIV-Exposed Infants",
            facility_uids,
        )

    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            lbw_data = compute_lbw_kpi(period_df, facility_uids)

            period_row = {
                "period": period,
                "period_display": period_display,
                "value": lbw_data["lbw_rate"],
                "LBW Cases (<2500g)": lbw_data["lbw_count"],
                "Total Weighed Births": lbw_data["total_weighed"],
            }

            for category_key in LBW_CATEGORIES.keys():
                period_row[f"{category_key}_rate"] = lbw_data["category_rates"][
                    category_key
                ]
                period_row[f"{category_key}_count"] = lbw_data["lbw_categories"][
                    category_key
                ]

            period_data.append(period_row)

        group = pd.DataFrame(period_data)
        render_lbw_trend_chart(
            group,
            "period_display",
            "value",
            "Low Birth Weight Rate (%)",
            bg_color,
            text_color,
            display_names,
            "LBW Cases (<2500g)",
            "Total Weighed Births",
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
    """Render comparison charts for both national and regional views"""

    kpi_config = get_kpi_config(kpi_selection)

    if comparison_mode == "facility":
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="PPH Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="PPH Cases",
                denominator_name="Total Deliveries",
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Uterotonic Administration Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="Uterotonic Cases",
                denominator_name="Total Deliveries",
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="ARV Prophylaxis Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="ARV Cases",
                denominator_name="HIV-Exposed Infants",
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Low Birth Weight Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="LBW Cases (<2500g)",
                denominator_name="Total Weighed Births",
            )
        else:
            # Use generic facility comparison for other KPIs
            render_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title=kpi_config.get("title", kpi_selection),
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=kpi_config.get("numerator_name", "Numerator"),
                denominator_name=kpi_config.get("denominator_name", "Denominator"),
            )

    else:  # region comparison (only for national)
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="PPH Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="PPH Cases",
                denominator_name="Total Deliveries",
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Uterotonic Administration Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="Uterotonic Cases",
                denominator_name="Total Deliveries",
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="ARV Prophylaxis Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="ARV Cases",
                denominator_name="HIV-Exposed Infants",
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Low Birth Weight Rate (%)",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="LBW Cases (<2500g)",
                denominator_name="Total Weighed Births",
            )
        else:
            # Use generic region comparison for other KPIs
            render_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title=kpi_config.get("title", kpi_selection),
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name=kpi_config.get("numerator_name", "Numerator"),
                denominator_name=kpi_config.get("denominator_name", "Denominator"),
            )


def render_additional_analytics(
    kpi_selection, filtered_events, facility_uids, bg_color, text_color
):
    """Render additional analytics charts - same in both files"""
    if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        render_obstetric_condition_pie_chart(
            filtered_events, facility_uids, bg_color, text_color
        )
    elif kpi_selection == "Delivered women who received uterotonic (%)":
        render_uterotonic_type_pie_chart(
            filtered_events, facility_uids, bg_color, text_color
        )
    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        render_lbw_category_pie_chart(
            filtered_events, facility_uids, bg_color, text_color
        )


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


# ========== UPDATED FILTER CONTROLS WITHOUT KPI SELECTION ==========


def render_simple_filter_controls(events_df, container=None):
    """Simple filter controls without KPI selection (KPI selection moved to tabs)"""
    if container is None:
        container = st

    filters = {}

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
        key="quick_range_simple",
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
                key="start_date_simple",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date_simple",
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
        key="period_label_simple",
    )

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key="bg_color_simple"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add a placeholder for kpi_selection to maintain compatibility
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi", "Institutional Maternal Death Rate (per 100,000 births)"
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

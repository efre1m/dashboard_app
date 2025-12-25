import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime

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
)

from utils.kpi_assisted import (
    compute_assisted_delivery_kpi,
    render_assisted_trend_chart,
    render_assisted_facility_comparison_chart,
    render_assisted_region_comparison_chart,
)

from utils.kpi_svd import (
    compute_svd_kpi,
    render_svd_trend_chart,
    render_svd_facility_comparison_chart,
    render_svd_region_comparison_chart,
)

from utils.kpi_missing_md import render_missing_md_simple_table


# KPI mapping for comparison charts - UPDATED NAMES
KPI_MAPPING = {
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
        "title": "IPPCAR (%)",
        "numerator_name": "FP Acceptances",
        "denominator_name": "Total Deliveries",
    },
    "Stillbirth Rate (%)": {
        "title": "Stillbirth Rate (%)",
        "numerator_name": "Stillbirths",
        "denominator_name": "Total Deliveries",
    },
    "Early Postnatal Care (PNC) Coverage (%)": {
        "title": "Early PNC Coverage (%)",
        "numerator_name": "Early PNC (≤48 hrs)",
        "denominator_name": "Total Deliveries",
    },
    "Institutional Maternal Death Rate (%)": {
        "title": "Maternal Death Rate (%)",
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
    "Missing Mode of Delivery": {
        "title": "Missing Mode of Delivery",
        "numerator_name": "Missing MD Cases",
        "denominator_name": "Total Deliveries",
    },
}


def get_text_color(bg_color):
    """Get auto text color for background"""
    return auto_text_color(bg_color)


def get_kpi_config(kpi_selection):
    """Get KPI configuration"""
    return KPI_MAPPING.get(kpi_selection, {})


# Common KPI options - UPDATED NAMES
KPI_OPTIONS = [
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "Stillbirth Rate (%)",
    "Early Postnatal Care (PNC) Coverage (%)",
    "Institutional Maternal Death Rate (%)",
    "C-Section Rate (%)",
    "Postpartum Hemorrhage (PPH) Rate (%)",
    "Delivered women who received uterotonic (%)",
    "ARV Prophylaxis Rate (%)",
    "Low Birth Weight (LBW) Rate (%)",
    "Assisted Delivery Rate (%)",
    "Normal Vaginal Delivery (SVD) Rate (%)",
    "Missing Mode of Delivery",
]


def render_kpi_tab_navigation():
    """Render professional tab navigation for KPI selection"""

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

    # KPI Grouping for Tab Navigation - UPDATED NAMES
    KPI_GROUPS = {
        "📉 Mortality": [
            "Institutional Maternal Death Rate (%)",
            "Stillbirth Rate (%)",
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
        "❓ Missing": [
            "Missing Mode of Delivery",
        ],
    }

    # Initialize session state for KPI selection
    if "selected_kpi" not in st.session_state:
        st.session_state.selected_kpi = "Institutional Maternal Death Rate (%)"

    # Create main KPI group tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 **Mortality**", "🚨 **Complications**", "🏥 **Care**", "❓ **Missing**"]
    )

    selected_kpi = st.session_state.selected_kpi

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 Maternal Death Rate",
                key="maternal_death_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Institutional Maternal Death Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Institutional Maternal Death Rate (%)"

        with col2:
            if st.button(
                "📊 Stillbirth Rate",
                key="stillbirth_btn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "Stillbirth Rate (%)" else "secondary"
                ),
            ):
                selected_kpi = "Stillbirth Rate (%)"

    with tab2:
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
        st.markdown("### Missing Data Analysis")
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
    """Get period column names from patient data"""
    period_col = None
    period_display_col = None
    period_sort_col = None

    # Look for period columns in the patient data
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
):
    """Render the trend chart with KPI-specific program stage dates - BOTH numerator and denominator filtered"""

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in patient_df.columns:
        # Try to find the facility ID column
        for col in patient_df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                patient_df = patient_df.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in patient_df.columns:
        st.error("❌ 'orgUnit' column not found in data. Cannot filter by UIDs.")
        return

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

    if patient_df.empty:
        st.info("⚠️ No data available for trend analysis.")
        return

    # Apply UID filter
    working_df = patient_df.copy()
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # Prepare data for trend chart using KPI-SPECIFIC program stage dates
    prepared_df, date_column_used = prepare_data_for_trend_chart(
        working_df, kpi_selection, facility_uids
    )

    if prepared_df.empty:
        # Show which date column was needed
        st.warning(
            f"⚠️ No data available for {kpi_selection} using its specific program stage date: '{date_column_used}'"
        )
        return

    # Ensure period columns exist with proper formatting
    if "period_display" not in prepared_df.columns:
        if "period" in prepared_df.columns:
            prepared_df["period_display"] = prepared_df["period"]
        else:
            if "event_date" in prepared_df.columns:
                prepared_df["period_display"] = (
                    prepared_df["event_date"].dt.strftime("%b-%y").str.capitalize()
                )  # Format as "Sep-25"
            else:
                prepared_df["period_display"] = "Period"

    if "period_sort" not in prepared_df.columns:
        if "period" in prepared_df.columns and prepared_df["period"].dtype == "object":
            try:
                prepared_df["period_sort"] = pd.to_datetime(
                    prepared_df["period_display"], format="%b-%y"
                )
            except:
                prepared_df["period_sort"] = prepared_df.index
        else:
            prepared_df["period_sort"] = prepared_df.index

    # Get unique periods in order
    unique_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Store the original filtered patient data for KPI calculations
    original_filtered_patients = working_df.copy()

    # Create period data - FILTER BOTH NUMERATOR AND DENOMINATOR BY KPI-SPECIFIC DATES
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period from prepared_df (already filtered by KPI-specific dates)
        period_df = prepared_df[prepared_df["period_display"] == period_display]

        if not period_df.empty:
            # Get TEI IDs from this period (using KPI-specific dates)
            period_tei_ids = period_df["tei_id"].dropna().unique()

            # Filter original data to get ALL rows for these TEI IDs
            # BUT only include rows that have the KPI-specific date
            if (
                date_column_used
                and date_column_used in original_filtered_patients.columns
            ):
                # Create a copy and filter by the KPI-specific date column
                period_patient_data = original_filtered_patients.copy()
                period_patient_data[date_column_used] = pd.to_datetime(
                    period_patient_data[date_column_used], errors="coerce"
                )
                # Keep only rows that have the KPI-specific date AND are in our TEI list
                period_patient_data = period_patient_data[
                    period_patient_data["tei_id"].isin(period_tei_ids)
                    & period_patient_data[date_column_used].notna()
                ].copy()
            else:
                # Fallback: just filter by TEI IDs
                period_patient_data = original_filtered_patients[
                    original_filtered_patients["tei_id"].isin(period_tei_ids)
                ].copy()

            # Now compute KPI using date-filtered data
            numerator, denominator, _ = get_numerator_denominator_for_kpi(
                period_patient_data, kpi_selection, facility_uids
            )

            # Calculate value
            value = (numerator / denominator * 100) if denominator > 0 else 0

            period_data.append(
                {
                    "period": period_display,
                    "period_display": period_display,
                    "period_sort": period_sort,
                    "value": value,
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                }
            )

    if not period_data:
        st.info("⚠️ No period data available for chart.")
        return

    # Create DataFrame
    group = pd.DataFrame(period_data)
    group = group.sort_values("period_sort")

    # Render the chart WITH TABLE (using KPI-specific dates)
    try:
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
            )
        else:
            # For all other KPIs, use the standard trend chart WITH TABLE
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
            )
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")


def render_comparison_chart(
    kpi_selection,
    patient_df=None,
    comparison_mode="facility",
    display_names=None,
    facility_uids=None,
    facilities_by_region=None,
    region_names=None,
    bg_color="#FFFFFF",
    text_color=None,
    is_national=False,
    filtered_patients=None,
):
    """Render comparison charts for both national and regional views WITH TABLES using KPI-specific dates"""

    df_to_use = filtered_patients if filtered_patients is not None else patient_df

    if df_to_use is None or df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    if kpi_selection == "Missing Mode of Delivery":
        st.info("⚠️ Comparison chart not available for Missing Mode of Delivery.")
        return

    kpi_config = get_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)

    # STANDARDIZE COLUMN NAMES - Ensure we have orgUnit
    if "orgUnit" not in df_to_use.columns:
        # Try to find the facility ID column
        for col in df_to_use.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df_to_use = df_to_use.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in df_to_use.columns:
        st.error("❌ Facility identifier column not found. Cannot perform comparison.")
        return

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_kpi(kpi_selection)

    # Prepare data using prepare_data_for_trend_chart with KPI-SPECIFIC program stage dates
    prepared_df, date_column_used = prepare_data_for_trend_chart(
        df_to_use, kpi_selection, facility_uids
    )

    if prepared_df.empty:
        # Show which date column was needed
        st.warning(
            f"⚠️ No data available for {kpi_selection} comparison using its specific program stage date: '{date_column}'"
        )
        return

    # Store the ORIGINAL patient data for KPI calculations
    original_patients = df_to_use.copy()

    # Create comparison data - STANDARDIZED COLUMNS
    comparison_data = []

    # Get all unique periods with proper formatting (using KPI-specific dates)
    if "period_display" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            prepared_df["period_display"] = (
                prepared_df["event_date"].dt.strftime("%b-%y").str.capitalize()
            )  # Format as "Sep-25"
        else:
            prepared_df["period_display"] = "All"

    # Ensure period_sort exists for sorting
    if "period_sort" not in prepared_df.columns and "event_date" in prepared_df.columns:
        prepared_df["period_sort"] = prepared_df["event_date"].dt.strftime("%Y%m")

    all_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_list = all_periods["period_display"].tolist()

    if comparison_mode == "facility":
        for period in period_list:
            # Get TEI IDs for this period from prepared_df (using KPI-specific dates)
            period_df = prepared_df[prepared_df["period_display"] == period]
            period_tei_ids = period_df["tei_id"].dropna().unique()

            for facility_uid, facility_name in zip(facility_uids, display_names):
                # Get ALL data for this facility from original dataset
                facility_data = original_patients[
                    original_patients["orgUnit"] == facility_uid
                ].copy()

                if not facility_data.empty:
                    # Filter to only patients who have events in this period (KPI-specific)
                    # AND have the KPI-specific date
                    if date_column_used and date_column_used in facility_data.columns:
                        # Filter by KPI-specific date column
                        facility_data[date_column_used] = pd.to_datetime(
                            facility_data[date_column_used], errors="coerce"
                        )
                        # Get TEI IDs that have the KPI-specific date
                        facility_tei_ids_with_date = facility_data[
                            facility_data[date_column_used].notna()
                        ]["tei_id"].unique()

                        # Intersection with period TEI IDs
                        facility_period_tei_ids = set(period_tei_ids) & set(
                            facility_tei_ids_with_date
                        )
                    else:
                        # Fallback: just use period TEI IDs
                        facility_period_tei_ids = set(period_tei_ids) & set(
                            facility_data["tei_id"].unique()
                        )

                    if facility_period_tei_ids:
                        # Get ALL data for these patients, filtered by KPI-specific date
                        facility_period_data = facility_data.copy()

                        # Filter by KPI-specific date
                        if (
                            date_column_used
                            and date_column_used in facility_period_data.columns
                        ):
                            facility_period_data[date_column_used] = pd.to_datetime(
                                facility_period_data[date_column_used], errors="coerce"
                            )
                            facility_period_data = facility_period_data[
                                facility_period_data["tei_id"].isin(
                                    facility_period_tei_ids
                                )
                                & facility_period_data[date_column_used].notna()
                            ].copy()
                        else:
                            facility_period_data = facility_period_data[
                                facility_period_data["tei_id"].isin(
                                    facility_period_tei_ids
                                )
                            ].copy()

                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            facility_period_data, kpi_selection, [facility_uid]
                        )

                        # Calculate value
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                        comparison_data.append(
                            {
                                "period_display": period,
                                "orgUnit": facility_uid,
                                "orgUnit_name": facility_name,  # Use orgUnit_name instead of Facility
                                "value": value,
                                "numerator": int(numerator),
                                "denominator": int(denominator),
                                "patient_count": len(facility_period_tei_ids),
                            }
                        )
                    else:
                        # Add entry with zero value
                        comparison_data.append(
                            {
                                "period_display": period,
                                "orgUnit": facility_uid,
                                "orgUnit_name": facility_name,
                                "value": 0,
                                "numerator": 0,
                                "denominator": 0,
                                "patient_count": 0,
                            }
                        )
                else:
                    # Add entry with zero value for missing facilities
                    comparison_data.append(
                        {
                            "period_display": period,
                            "orgUnit": facility_uid,
                            "orgUnit_name": facility_name,
                            "value": 0,
                            "numerator": 0,
                            "denominator": 0,
                            "patient_count": 0,
                        }
                    )

        if not comparison_data:
            st.info("⚠️ No comparison data available.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # FIX: STANDARDIZE COLUMN NAMES FOR CHART FUNCTION
        # Ensure we have the expected column names for the chart function
        if (
            "orgUnit_name" in comparison_df.columns
            and "Facility" not in comparison_df.columns
        ):
            comparison_df = comparison_df.rename(columns={"orgUnit_name": "Facility"})

        # If still no Facility column, create one from display_names
        if "Facility" not in comparison_df.columns and display_names:
            # Create a mapping of orgUnit to facility name
            facility_name_map = {}
            for uid, name in zip(facility_uids, display_names):
                facility_name_map[uid] = name

            # Add Facility column based on mapping
            comparison_df["Facility"] = comparison_df["orgUnit"].map(facility_name_map)

        # STANDARDIZE: Ensure chart functions get consistent data
        display_names_from_data = (
            comparison_df["Facility"].unique().tolist()
            if "Facility" in comparison_df.columns
            else []
        )
        facility_uids_from_data = comparison_df["orgUnit"].unique().tolist()

        # Check if we have the required columns
        if "Facility" not in comparison_df.columns:
            st.error(
                f"❌ Could not create Facility column. Available columns: {list(comparison_df.columns)}"
            )
            return

        # Call the appropriate chart function WITH TABLE
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Assisted Delivery Rate (%)":
            render_assisted_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
            render_svd_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        else:
            render_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names_from_data,
                facility_uids=facility_uids_from_data,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )

    elif comparison_mode == "region" and is_national:
        # Region comparison (only for national view)
        if not region_names or not facilities_by_region:
            st.error("❌ Region data not provided for regional comparison.")
            return

        # Prepare region comparison data
        region_data = []

        # Get facility UIDs for each region
        region_facility_mapping = {}
        for region_name in region_names:
            region_facility_mapping[region_name] = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

        for period in period_list:
            # Get TEI IDs for this period from prepared_df (KPI-specific dates)
            period_df = prepared_df[prepared_df["period_display"] == period]
            period_tei_ids = period_df["tei_id"].dropna().unique()

            for region_name in region_names:
                region_facility_uids = region_facility_mapping.get(region_name, [])
                if not region_facility_uids:
                    continue

                # Get ALL data for this region from original dataset
                region_data_all = original_patients[
                    original_patients["orgUnit"].isin(region_facility_uids)
                ].copy()

                if not region_data_all.empty:
                    # Filter to only patients who have events in this period (KPI-specific)
                    # AND have the KPI-specific date
                    if date_column_used and date_column_used in region_data_all.columns:
                        # Filter by KPI-specific date column
                        region_data_all[date_column_used] = pd.to_datetime(
                            region_data_all[date_column_used], errors="coerce"
                        )
                        # Get TEI IDs that have the KPI-specific date
                        region_tei_ids_with_date = region_data_all[
                            region_data_all[date_column_used].notna()
                        ]["tei_id"].unique()

                        # Intersection with period TEI IDs
                        region_period_tei_ids = set(period_tei_ids) & set(
                            region_tei_ids_with_date
                        )
                    else:
                        # Fallback: just use period TEI IDs
                        region_period_tei_ids = set(period_tei_ids) & set(
                            region_data_all["tei_id"].unique()
                        )

                    if region_period_tei_ids:
                        # Get ALL data for these patients, filtered by KPI-specific date
                        region_period_data = region_data_all.copy()

                        # Filter by KPI-specific date
                        if (
                            date_column_used
                            and date_column_used in region_period_data.columns
                        ):
                            region_period_data[date_column_used] = pd.to_datetime(
                                region_period_data[date_column_used], errors="coerce"
                            )
                            region_period_data = region_period_data[
                                region_period_data["tei_id"].isin(region_period_tei_ids)
                                & region_period_data[date_column_used].notna()
                            ].copy()
                        else:
                            region_period_data = region_period_data[
                                region_period_data["tei_id"].isin(region_period_tei_ids)
                            ].copy()

                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            region_period_data, kpi_selection, region_facility_uids
                        )

                        # Calculate value
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                        region_data.append(
                            {
                                "period_display": period,
                                "Region": region_name,
                                "value": value,
                                "numerator": int(numerator),
                                "denominator": int(denominator),
                                "patient_count": len(region_period_tei_ids),
                            }
                        )
                    else:
                        # Add zero entry
                        region_data.append(
                            {
                                "period_display": period,
                                "Region": region_name,
                                "value": 0,
                                "numerator": 0,
                                "denominator": 0,
                                "patient_count": 0,
                            }
                        )
                else:
                    # Add zero entry
                    region_data.append(
                        {
                            "period_display": period,
                            "Region": region_name,
                            "value": 0,
                            "numerator": 0,
                            "denominator": 0,
                            "patient_count": 0,
                        }
                    )

        if not region_data:
            st.info("⚠️ No comparison data available for regions.")
            return

        region_df = pd.DataFrame(region_data)

        # Call the appropriate region comparison function WITH TABLE
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            render_pph_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            render_uterotonic_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            render_arv_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
            render_lbw_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Assisted Delivery Rate (%)":
            render_assisted_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        elif kpi_selection == "Normal Vaginal Delivery (SVD) Rate (%)":
            render_svd_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
        else:
            render_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
    else:
        if comparison_mode == "region":
            st.info(
                "⚠️ Region comparison is only available in national view when region data is provided."
            )
        else:
            st.info("⚠️ Invalid comparison mode selected.")


def render_additional_analytics(
    kpi_selection, patient_df, facility_uids, bg_color, text_color
):
    """Render additional analytics charts"""
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


def normalize_patient_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single datetime column 'event_date' exists for patient data"""
    if df.empty:
        return df

    df = df.copy()

    # Look for program stage event dates first
    program_stage_date_columns = [
        col
        for col in df.columns
        if col.startswith("event_date_") or col == "event_date"
    ]

    for col in program_stage_date_columns:
        try:
            df["event_date"] = pd.to_datetime(df[col], errors="coerce")
            if not df["event_date"].isna().all():
                break
        except:
            continue

    # If no program stage date found, use current date
    if "event_date" not in df.columns or df["event_date"].isna().all():
        df["event_date"] = pd.Timestamp.now().normalize()

    return df


def render_patient_filter_controls(patient_df, container=None, context="default"):
    """Simple filter controls for patient data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
    key_suffix = f"_{context}"

    # Time Period options
    time_options = [
        "All Time",
        "Custom Range",
        "Today",
        "This Week",
        "Last Week",
        "This Month",
        "Last Month",
        "This Year",
        "Last Year",
    ]

    # Get current selection
    current_selection = st.session_state.get(f"quick_range{key_suffix}", "All Time")

    # Ensure the value is valid
    if current_selection not in time_options:
        current_selection = "All Time"

    filters["quick_range"] = container.selectbox(
        "📅 Time Period",
        time_options,
        index=time_options.index(current_selection),
        key=f"quick_range{key_suffix}",
    )

    # Get REAL VALID dates from patient dataframe
    min_date, max_date = _get_patient_date_range(patient_df)

    # Ensure min_date is not earlier than 2000
    if min_date < datetime.date(2000, 1, 1):
        min_date = datetime.date(2000, 1, 1)

    # Ensure max_date is not later than 2035
    if max_date > datetime.date(2035, 12, 31):
        max_date = datetime.date(2035, 12, 31)

    # Set dates based on selection
    if filters["quick_range"] == "All Time":
        filters["start_date"] = min_date
        filters["end_date"] = max_date

    elif filters["quick_range"] == "Custom Range":
        # For Custom Range, use CURRENT YEAR as default
        today = datetime.date.today()
        default_start = datetime.date(today.year, 1, 1)
        default_end = datetime.date(today.year, 12, 31)

        # Ensure defaults are within min/max bounds
        if default_start < min_date:
            default_start = min_date
        if default_end > max_date:
            default_end = max_date

        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=default_start,
                min_value=datetime.date(2000, 1, 1),
                max_value=max_date,
                key=f"start_date{key_suffix}",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=datetime.date(2035, 12, 31),
                key=f"end_date{key_suffix}",
            )
    else:
        temp_df = patient_df.copy()
        _df_for_dates = temp_df.copy()
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level
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

    # Save to session state
    st.session_state.period_label = filters["period_label"]

    # Also save to filters dictionary in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters["period_label"] = filters["period_label"]
    st.session_state.filters["start_date"] = filters["start_date"]
    st.session_state.filters["end_date"] = filters["end_date"]

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add placeholder for kpi_selection
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi", "Institutional Maternal Death Rate (%)"
    )

    return filters


def _get_patient_date_range(patient_df):
    """Get REAL min/max dates from patient dataframe"""
    import datetime
    import pandas as pd

    if patient_df.empty:
        today = datetime.date.today()
        return today, today

    # Collect ALL VALID dates from ALL possible date columns
    all_valid_dates = []

    # Check ALL columns that might contain dates
    for col in patient_df.columns:
        if "date" in col.lower() or "Date" in col:
            try:
                # Try to convert to datetime
                dates = pd.to_datetime(patient_df[col], errors="coerce")
                # Drop NaT (invalid dates)
                valid_dates = dates.dropna()

                if not valid_dates.empty:
                    # Filter out dates that are clearly invalid
                    for date_val in valid_dates:
                        try:
                            date_dt = date_val.to_pydatetime()
                            year = date_dt.year
                            # Only accept dates from year 2000 onward
                            if year >= 2000 and year <= 2030:
                                all_valid_dates.append(date_val)
                        except:
                            continue
            except Exception:
                continue

    if all_valid_dates:
        # Find the REAL minimum and maximum VALID dates
        min_date = min(all_valid_dates)
        max_date = max(all_valid_dates)

        # Convert to date objects
        if hasattr(min_date, "date"):
            min_date = min_date.date()
        if hasattr(max_date, "date"):
            max_date = max_date.date()

        return min_date, max_date

    # If no valid dates found, use current year
    today = datetime.date.today()
    current_year_start = datetime.date(today.year, 1, 1)
    current_year_end = datetime.date(today.year, 12, 31)

    return current_year_start, current_year_end


def apply_patient_filters(patient_df, filters, facility_uids=None):
    """Apply filters to patient dataframe"""
    if patient_df.empty:
        return patient_df

    df = patient_df.copy()

    # For "All Time", don't apply any date filtering at all!
    quick_range = filters.get("quick_range", "")
    skip_date_filtering = quick_range == "All Time"

    # STANDARDIZE COLUMN NAMES - Ensure we have orgUnit
    if "orgUnit" not in df.columns:
        # Try to find the facility ID column
        for col in df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df = df.rename(columns={col: "orgUnit"})
                break

    # Apply facility filter FIRST using UIDs if orgUnit exists
    if facility_uids and "orgUnit" in df.columns:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        facility_mask = df["orgUnit"].isin(facility_uids)
        df = df[facility_mask].copy()

    # Check for 'combined_date' first (from regional.py)
    if "combined_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["combined_date"], errors="coerce")
    elif "event_date" not in df.columns:
        # Normalize dates if no event_date column
        df = normalize_patient_dates(df)

    # Only apply date filtering if NOT "All Time"
    if (
        not skip_date_filtering
        and filters.get("start_date")
        and filters.get("end_date")
    ):
        try:
            # Convert filter dates to datetime
            start_dt = pd.Timestamp(filters["start_date"])
            end_dt = pd.Timestamp(filters["end_date"])

            # Filter by date range
            date_mask = (df["event_date"] >= start_dt) & (df["event_date"] <= end_dt)
            df = df[date_mask].copy()
        except Exception:
            pass

    # Ensure period_label is in filters
    if "period_label" not in filters:
        if "period_label" in st.session_state:
            filters["period_label"] = st.session_state.period_label
        else:
            filters["period_label"] = "Monthly"

    # Save to session state for prepare_data_for_trend_chart
    st.session_state.period_label = filters["period_label"]

    # Also ensure filters dict is in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters.update(filters)

    # Assign period AFTER date filtering
    try:
        valid_date_rows = df["event_date"].notna().sum()
        if valid_date_rows > 0:
            df = assign_period(df, "event_date", filters["period_label"])
        else:
            # Create empty period columns
            df["period_display"] = ""
            df["period_sort"] = 0
    except Exception:
        # Create empty period columns as fallback
        df["period_display"] = ""
        df["period_sort"] = 0

    # Make sure orgUnit is preserved
    if "orgUnit" not in df.columns and "orgUnit" in patient_df.columns:
        # Try to add it back using tei_id
        if "tei_id" in df.columns and "tei_id" in patient_df.columns:
            tei_ids = df["tei_id"].unique()
            # Get orgUnit from original data for these TEI IDs
            orgunit_mapping = patient_df[["tei_id", "orgUnit"]].drop_duplicates()
            df = pd.merge(df, orgunit_mapping, on="tei_id", how="left")

    return df


def get_period_data_for_kpi(kpi_selection, patient_df, facility_uids):
    """Get period-based data for a specific KPI from patient data"""
    date_column = get_relevant_date_column_for_kpi(kpi_selection)

    # Extract period columns using program stage specific date
    df_with_periods = extract_period_columns(patient_df, date_column)

    if df_with_periods.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df_with_periods = df_with_periods[
            df_with_periods["orgUnit"].isin(facility_uids)
        ]

    # Sort by period
    if "period_sort" in df_with_periods.columns:
        df_with_periods = df_with_periods.sort_values("period_sort")

    return df_with_periods


def compute_kpi_for_period(kpi_selection, period_df, facility_uids):
    """Compute KPI for a specific period using patient data"""
    if period_df.empty:
        return _get_default_kpi_data(kpi_selection)

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
    elif kpi_selection == "Stillbirth Rate (%)":
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
    elif kpi_selection == "Institutional Maternal Death Rate (%)":
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


def _get_default_kpi_data(kpi_selection):
    """Get default empty data for different KPI types"""
    defaults = {
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
            "ippcar": 0,
            "fp_acceptance": 0,
            "total_deliveries": 0,
        },
        "Stillbirth Rate (%)": {
            "stillbirth_rate": 0,
            "stillbirths": 0,
            "total_deliveries_sb": 0,
        },
        "Early Postnatal Care (PNC) Coverage (%)": {
            "pnc_coverage": 0,
            "early_pnc": 0,
            "total_deliveries": 0,
        },
        "Institutional Maternal Death Rate (%)": {
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


def get_date_column_from_patient_df(df, kpi_selection):
    """Get the appropriate date column from patient-level data based on KPI - USE SPECIFIC PROGRAM STAGE"""
    # Use the same mapping as in kpi_utils
    date_column = get_relevant_date_column_for_kpi(kpi_selection)

    # Check if this specific column exists
    if date_column in df.columns:
        return date_column

    # If not, warn and don't use wrong dates
    st.warning(f"⚠️ Required date column '{date_column}' not found for {kpi_selection}")
    return None

# utils/dash_co.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime

# CLEANED imports from kpi_utils - only what's actually used:
from utils.kpi_utils import (
    compute_kpis,
    auto_text_color,
    prepare_data_for_trend_chart,
    extract_period_columns,
    get_relevant_date_column_for_kpi,
    get_numerator_denominator_for_kpi,
    get_combined_date_for_kpi,
    DELIVERY_DATE_COL,
    PNC_DATE_COL,
    DISCHARGE_DATE_COL,
    ENROLLMENT_DATE_COL,
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
    "Missing Mode of Delivery",
]


def render_kpi_tab_navigation():
    """Render professional tab navigation for KPI selection without duplication"""

    # Custom CSS for professional tab styling and active button highlighting
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
        "❓ Missing": [
            "Missing Mode of Delivery",
        ],
    }

    # Initialize session state for KPI selection
    if "selected_kpi" not in st.session_state:
        st.session_state.selected_kpi = (
            "Institutional Maternal Death Rate (per 100,000 births)"
        )

    # Create main KPI group tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 **Mortality**", "🚨 **Complications**", "🏥 **Care**", "❓ **Missing**"]
    )

    selected_kpi = st.session_state.selected_kpi

    with tab1:
        # Mortality KPIs
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
        # Complications KPIs
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
        # Care KPIs
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
        # Missing Data KPIs
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
    facility_uids,  # ✅ UIDs passed here
    display_names,
    bg_color,
    text_color,
    comparison_mode="overall",
    facilities_by_region=None,
    region_names=None,
    show_table=False,
):
    """Render the trend chart - UPDATED VERSION: Uses UIDs for filtering"""

    print(f"\n{'='*80}")
    print(f"🚨 START render_trend_chart_section for {kpi_selection}")
    print(f"{'='*80}")

    # ✅ DEBUG: Check what we're receiving
    print(f"📊 PARAMETERS RECEIVED:")
    print(f"   KPI Selection: {kpi_selection}")
    print(f"   Patient DF shape: {patient_df.shape}")
    print(f"   Facility UIDs: {facility_uids}")
    print(f"   Display Names: {display_names}")

    # Check if orgUnit column exists
    if "orgUnit" not in patient_df.columns:
        st.error("❌ 'orgUnit' column not found in data. Cannot filter by UIDs.")
        print(f"❌ ERROR: 'orgUnit' column missing!")
        print(
            f"   Available columns: {[col for col in patient_df.columns if 'org' in col.lower()]}"
        )
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

    # ✅ DEBUG: Show what we're working with
    print(f"\n🔍 [TREND CHART] {kpi_selection}")
    print(f"   Total rows: {len(patient_df)}")
    print(f"   Facility UIDs to filter: {facility_uids}")

    # ✅ Apply UID filter
    working_df = patient_df.copy()
    if facility_uids and "orgUnit" in working_df.columns:
        before_filter = len(working_df)
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()
        after_filter = len(working_df)
        print(
            f"   After UID filter: {after_filter} rows (removed {before_filter - after_filter})"
        )

        # Show UID breakdown
        for uid in facility_uids:
            uid_data = working_df[working_df["orgUnit"] == uid]
            if not uid_data.empty:
                facility_name = (
                    uid_data["orgUnit_name"].iloc[0]
                    if "orgUnit_name" in uid_data.columns
                    else "Unknown"
                )
                print(f"   UID {uid} ({facility_name}): {len(uid_data)} rows")

    if "tei_id" in working_df.columns:
        print(f"   Unique TEI IDs after filtering: {working_df['tei_id'].nunique()}")

    # ✅ Use the updated prepare_data_for_trend_chart which works with patient data
    print(f"\n📊 PREPARING DATA FOR TREND CHART...")
    prepared_df = prepare_data_for_trend_chart(working_df, kpi_selection, facility_uids)

    if prepared_df.empty:
        st.info("⚠️ No prepared data available for trend analysis.")
        return

    # Debug: Check what date source was used
    if "date_source" in prepared_df.columns:
        date_sources = prepared_df["date_source"].value_counts()
        print(f"📅 Date sources used: {date_sources.to_dict()}")

    # Ensure period columns exist
    if "period_display" not in prepared_df.columns:
        # Try to extract from period column
        if "period" in prepared_df.columns:
            prepared_df["period_display"] = prepared_df["period"]
            print(f"✅ Created period_display from 'period' column")
        else:
            # Try to create from event_date
            if "event_date" in prepared_df.columns:
                prepared_df["period_display"] = prepared_df["event_date"].dt.strftime(
                    "%b-%y"
                )
                print(f"✅ Created period_display from event_date")
            else:
                prepared_df["period_display"] = "Period"
                print(f"⚠️ Using default period_display")

    if "period_sort" not in prepared_df.columns:
        # Create sorting column
        if "period" in prepared_df.columns and prepared_df["period"].dtype == "object":
            # Try to extract month-year for sorting
            try:
                prepared_df["period_sort"] = pd.to_datetime(
                    prepared_df["period_display"], format="%b-%y"
                )
            except:
                prepared_df["period_sort"] = prepared_df.index
        else:
            prepared_df["period_sort"] = prepared_df.index
        print(f"✅ Created period_sort column")

    # Get unique periods in order
    unique_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    print(f"\n📊 PERIOD ANALYSIS:")
    print(f"   Found {len(unique_periods)} unique periods")
    print(f"   Periods: {unique_periods['period_display'].tolist()}")

    # ✅ CRITICAL FIX: We need the ORIGINAL patient_df (not prepared_df) for KPI calculations
    # Store the original filtered patient data (with UID filter already applied)
    original_filtered_patients = working_df.copy()
    print(
        f"   Original filtered patients for KPI calculation: {len(original_filtered_patients)} rows"
    )
    print(
        f"   Original unique TEI IDs: {original_filtered_patients['tei_id'].nunique()}"
    )
    print(
        f"   Original unique orgUnits: {original_filtered_patients['orgUnit'].nunique()}"
    )

    # Create period data
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period from prepared_df (for period assignment)
        period_df = prepared_df[prepared_df["period_display"] == period_display]

        print(f"\n   📅 Processing period: {period_display}")
        print(f"      Rows in period from prepared_df: {len(period_df)}")

        if not period_df.empty:
            # ✅ Get TEI IDs from this period
            period_tei_ids = period_df["tei_id"].dropna().unique()
            print(f"      Unique TEI IDs in period: {len(period_tei_ids)}")

            # ✅ CRITICAL: Get ALL data for these TEI IDs from original dataset
            # This ensures we include ALL data for patients in this period
            period_patient_data = original_filtered_patients[
                original_filtered_patients["tei_id"].isin(period_tei_ids)
            ].copy()

            print(
                f"      Total patient data for period (all events): {len(period_patient_data)} rows"
            )
            print(
                f"      Unique patients in period data: {period_patient_data['tei_id'].nunique()}"
            )
            print(
                f"      Unique orgUnits in period data: {period_patient_data['orgUnit'].nunique()}"
            )

            # ✅ Use ALL patient data for this period for KPI calculation
            numerator, denominator, _ = get_numerator_denominator_for_kpi(
                period_patient_data, kpi_selection, facility_uids
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
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                }
            )

            # Debug output for each period
            print(
                f"      📊 Period {period_display}: {numerator}/{denominator} = {value:.2f}"
            )

            # Check if we're missing patients
            if denominator < len(period_tei_ids):
                print(
                    f"      ⚠️ WARNING: Denominator ({denominator}) < Patients in period ({len(period_tei_ids)})"
                )
        else:
            print(f"      ❌ No data for period {period_display}")

    if not period_data:
        st.info("⚠️ No period data available for chart.")
        return

    # Create DataFrame
    group = pd.DataFrame(period_data)
    group = group.sort_values("period_sort")

    print(f"\n✅ TREND DATA SUMMARY:")
    print(f"   Periods in chart: {len(group)}")
    print(f"   Data sample:")
    print(group.head())

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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
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
                facility_uids,  # ✅ Pass UIDs
                show_table=show_table,
            )
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")
        import traceback

        st.error(traceback.format_exc())

    print(f"{'='*80}")
    print(f"✅ END render_trend_chart_section for {kpi_selection}")
    print(f"{'='*80}")


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
                numerator_label,
                denominator_label,
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
                numerator_label,
                denominator_label,
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
                numerator_label,
                denominator_label,
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
                numerator_label,
                denominator_label,
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
                numerator_label,
                denominator_label,
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
    facility_uids=None,  # ✅ UIDs passed here
    facilities_by_region=None,
    region_names=None,
    bg_color="#FFFFFF",
    text_color=None,
    is_national=False,
    filtered_patients=None,
    show_table=False,
):
    """Render comparison charts for both national and regional views - UPDATED for UID filtering"""

    print(f"\n{'='*80}")
    print(f"🚨 START render_comparison_chart for {kpi_selection}")
    print(f"{'='*80}")

    # USE filtered_patients IF PROVIDED, OTHERWISE USE patient_df
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

    # ✅ CRITICAL: Check if orgUnit column exists in the data
    if "orgUnit" not in df_to_use.columns:
        st.error(
            "❌ Column 'orgUnit' not found in the data. Cannot perform comparison."
        )
        st.info(
            "Please ensure your data contains an 'orgUnit' column with facility IDs."
        )
        return

    print(f"\n🔍 RENDER_COMPARISON_CHART - UID MODE:")
    print(f"   KPI: {kpi_selection}")
    print(f"   Total rows: {len(df_to_use)}")
    print(f"   Facility UIDs: {facility_uids}")
    print(f"   Display names: {display_names}")
    print(f"   Comparison mode: {comparison_mode}")
    print(f"   Is national: {is_national}")

    # ✅ Prepare data using the updated prepare_data_for_trend_chart for patient data
    prepared_df = prepare_data_for_trend_chart(df_to_use, kpi_selection, facility_uids)

    if prepared_df.empty:
        st.info("⚠️ No prepared data available for comparison.")
        return

    # ✅ CRITICAL FIX: Store the ORIGINAL patient data for KPI calculations
    original_patients = df_to_use.copy()
    print(f"\n📊 COMPARISON CHART - PATIENT DATA:")
    print(f"   Original patient data: {len(original_patients)} rows")
    print(
        f"   Unique TEI IDs in original data: {original_patients['tei_id'].nunique()}"
    )
    print(f"   Unique orgUnits: {original_patients['orgUnit'].nunique()}")

    # Create comparison data
    comparison_data = []

    # Get all unique periods
    if "period_display" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            prepared_df["period_display"] = prepared_df["event_date"].dt.strftime(
                "%b-%y"
            )
            print(f"✅ Created period_display from event_date")
        elif ENROLLMENT_DATE_COL in prepared_df.columns:
            prepared_df["period_display"] = prepared_df[
                ENROLLMENT_DATE_COL
            ].dt.strftime("%b-%y")
            print(f"✅ Created period_display from enrollment_date")
        else:
            prepared_df["period_display"] = "All"
            print(f"⚠️ Using default period_display")

    all_periods = prepared_df["period_display"].unique()
    print(f"\n📊 PERIODS FOR COMPARISON:")
    print(f"   Found {len(all_periods)} periods: {list(all_periods)}")

    if comparison_mode == "facility":
        print(f"\n🔧 FACILITY COMPARISON MODE (USING UIDs):")
        print(f"   Number of facilities: {len(facility_uids)}")

        # Validate UID mapping
        for facility_uid, facility_name in zip(facility_uids, display_names):
            facility_data = original_patients[
                original_patients["orgUnit"] == facility_uid
            ]
            if not facility_data.empty:
                actual_name = (
                    facility_data["orgUnit_name"].iloc[0]
                    if "orgUnit_name" in facility_data.columns
                    else "Unknown"
                )
                print(
                    f"   UID {facility_uid} mapped to: {actual_name} (expected: {facility_name})"
                )
            else:
                print(
                    f"   ⚠️ UID {facility_uid} not found in data (expected: {facility_name})"
                )

        # Facility comparison
        for period in all_periods:
            print(f"\n   📅 Processing period: {period}")

            # Get TEI IDs for this period from prepared_df
            period_df = prepared_df[prepared_df["period_display"] == period]
            period_tei_ids = period_df["tei_id"].dropna().unique()

            print(f"      Unique TEI IDs in period: {len(period_tei_ids)}")

            for facility_uid, facility_name in zip(facility_uids, display_names):
                # ✅ Get ALL data for this facility from original dataset
                facility_data = original_patients[
                    original_patients["orgUnit"] == facility_uid
                ].copy()

                if not facility_data.empty:
                    # Filter to only patients who have events in this period
                    facility_period_tei_ids = set(period_tei_ids) & set(
                        facility_data["tei_id"].unique()
                    )

                    if facility_period_tei_ids:
                        # Get ALL data for these patients
                        facility_period_data = facility_data[
                            facility_data["tei_id"].isin(facility_period_tei_ids)
                        ].copy()

                        print(
                            f"      ✅ Facility {facility_name} ({facility_uid}): {len(facility_period_data)} rows, {len(facility_period_tei_ids)} patients"
                        )

                        # ✅ Use ALL patient data for KPI calculation
                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            facility_period_data, kpi_selection, [facility_uid]
                        )

                        # Calculate value based on KPI type
                        if "IPPCAR" in kpi_selection:
                            value = (
                                (numerator / denominator * 100)
                                if denominator > 0
                                else 0
                            )
                        elif "Stillbirth Rate" in kpi_selection:
                            value = (
                                (numerator / denominator * 1000)
                                if denominator > 0
                                else 0
                            )
                        elif "Maternal Death Rate" in kpi_selection:
                            value = (
                                (numerator / denominator * 100000)
                                if denominator > 0
                                else 0
                            )
                        elif "%" in kpi_selection:
                            value = (
                                (numerator / denominator * 100)
                                if denominator > 0
                                else 0
                            )
                        else:
                            value = 0

                        comparison_data.append(
                            {
                                "period_display": period,
                                "Facility": facility_name,
                                "facility_uid": facility_uid,  # ✅ Store UID for reference
                                "value": value,
                                "numerator": int(numerator),
                                "denominator": int(denominator),
                                "patient_count": len(facility_period_tei_ids),
                            }
                        )
                    else:
                        print(
                            f"      ⚠️ Facility {facility_name} ({facility_uid}): No patients in this period"
                        )

                        # Still add entry with zero value
                        comparison_data.append(
                            {
                                "period_display": period,
                                "Facility": facility_name,
                                "facility_uid": facility_uid,
                                "value": 0,
                                "numerator": 0,
                                "denominator": 0,
                                "patient_count": 0,
                            }
                        )
                else:
                    print(
                        f"      ❌ Facility {facility_name} ({facility_uid}): No data found for UID"
                    )

                    # Add entry with zero value for missing facilities
                    comparison_data.append(
                        {
                            "period_display": period,
                            "Facility": facility_name,
                            "facility_uid": facility_uid,
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

        print(f"\n✅ COMPARISON DATA SUMMARY:")
        print(f"   Total comparison rows: {len(comparison_df)}")
        print(
            f"   Unique facilities in comparison: {comparison_df['Facility'].nunique()}"
        )
        print(
            f"   Unique periods in comparison: {comparison_df['period_display'].nunique()}"
        )

        # Show sample comparison data
        print(f"\n📋 SAMPLE COMPARISON DATA:")
        print(comparison_df.head())

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

    elif comparison_mode == "region" and is_national:
        # Region comparison (only for national view)
        if not region_names or not facilities_by_region:
            st.error("❌ Region data not provided for regional comparison.")
            return

        print(f"\n🔧 REGION COMPARISON MODE:")
        print(f"   Number of regions: {len(region_names)}")

        # Prepare region comparison data
        region_data = []

        # Get facility UIDs for each region
        region_facility_mapping = {}
        for region_name in region_names:
            region_facility_mapping[region_name] = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            print(
                f"   Region {region_name}: {len(region_facility_mapping[region_name])} facilities"
            )

        for period in all_periods:
            print(f"\n   📅 Processing period: {period}")

            # Get TEI IDs for this period from prepared_df
            period_df = prepared_df[prepared_df["period_display"] == period]
            period_tei_ids = period_df["tei_id"].dropna().unique()

            print(f"      Unique TEI IDs in period: {len(period_tei_ids)}")

            for region_name in region_names:
                region_facility_uids = region_facility_mapping.get(region_name, [])
                if not region_facility_uids:
                    print(f"      Region {region_name}: No facilities")
                    continue

                # ✅ Get ALL data for this region from original dataset
                region_data_all = original_patients[
                    original_patients["orgUnit"].isin(region_facility_uids)
                ].copy()

                if not region_data_all.empty:
                    # Filter to only patients who have events in this period
                    region_period_tei_ids = set(period_tei_ids) & set(
                        region_data_all["tei_id"].unique()
                    )

                    if region_period_tei_ids:
                        # Get ALL data for these patients
                        region_period_data = region_data_all[
                            region_data_all["tei_id"].isin(region_period_tei_ids)
                        ].copy()

                        print(
                            f"      Region {region_name}: {len(region_period_data)} rows, {len(region_period_tei_ids)} patients"
                        )

                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            region_period_data, kpi_selection, region_facility_uids
                        )

                        # Calculate value based on KPI type
                        if "IPPCAR" in kpi_selection:
                            value = (
                                (numerator / denominator * 100)
                                if denominator > 0
                                else 0
                            )
                        elif "Stillbirth Rate" in kpi_selection:
                            value = (
                                (numerator / denominator * 1000)
                                if denominator > 0
                                else 0
                            )
                        elif "Maternal Death Rate" in kpi_selection:
                            value = (
                                (numerator / denominator * 100000)
                                if denominator > 0
                                else 0
                            )
                        elif "%" in kpi_selection:
                            value = (
                                (numerator / denominator * 100)
                                if denominator > 0
                                else 0
                            )
                        else:
                            value = 0

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
                        print(f"      Region {region_name}: No patients in this period")

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
                    print(f"      Region {region_name}: No data found")

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

        print(f"\n✅ REGION COMPARISON DATA SUMMARY:")
        print(f"   Total region comparison rows: {len(region_df)}")
        print(f"   Unique regions in comparison: {region_df['Region'].nunique()}")
        print(
            f"   Unique periods in comparison: {region_df['period_display'].nunique()}"
        )

        # Call the appropriate region comparison function
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
                show_table=show_table,
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
                show_table=show_table,
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
                show_table=show_table,
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
                show_table=show_table,
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
                show_table=show_table,
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
                show_table=show_table,
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
                show_table=show_table,
            )
    else:
        if comparison_mode == "region":
            st.info(
                "⚠️ Region comparison is only available in national view when region data is provided."
            )
        else:
            st.info("⚠️ Invalid comparison mode selected.")

    print(f"{'='*80}")
    print(f"✅ END render_comparison_chart for {kpi_selection}")
    print(f"{'='*80}")


def render_additional_analytics(
    kpi_selection, patient_df, facility_uids, bg_color, text_color
):
    """Render additional analytics charts - UPDATED for patient data"""
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

    # For patient data, look for date columns in order of preference
    date_preference_order = [
        "event_date_delivery_summary",  # Most specific for maternal KPIs
        "event_date_postpartum_care",  # For PNC-related KPIs
        ENROLLMENT_DATE_COL,  # Enrollment date fallback
        DELIVERY_DATE_COL,  # Delivery date column
        PNC_DATE_COL,  # PNC date column
        DISCHARGE_DATE_COL,  # Discharge date column
    ]

    # Add any other date columns that might exist
    for col in df.columns:
        if "date" in col.lower() and col not in date_preference_order:
            date_preference_order.append(col)

    for col in date_preference_order:
        if col in df.columns:
            try:
                # Try parsing ISO format
                df["event_date"] = pd.to_datetime(df[col], errors="coerce")
                # If that fails, try US format
                if df["event_date"].isna().all():
                    df["event_date"] = pd.to_datetime(
                        df[col], format="%m/%d/%Y", errors="coerce"
                    )
                if not df["event_date"].isna().all():
                    print(f"✅ Using date column: {col} for patient data")
                    break
            except:
                continue

    # If no date column found, use current date
    if "event_date" not in df.columns or df["event_date"].isna().all():
        df["event_date"] = pd.Timestamp.now().normalize()
        print("⚠️ No valid date columns found, using current date")

    return df


def normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure enrollment_date is datetime from various formats."""
    if df.empty or ENROLLMENT_DATE_COL not in df.columns:
        return df
    df = df.copy()
    df[ENROLLMENT_DATE_COL] = pd.to_datetime(df[ENROLLMENT_DATE_COL], errors="coerce")
    return df


def render_simple_filter_controls(patient_df, container=None, context="default"):
    """
    Simple filter controls for patient data without KPI selection.
    This is an alias for render_patient_filter_controls for backward compatibility.
    """
    return render_patient_filter_controls(patient_df, container, context)


def apply_simple_filters(patient_df, filters, facility_uids=None):
    """
    Apply filters to patient dataframe.
    This is an alias for apply_patient_filters for backward compatibility.
    """
    return apply_patient_filters(patient_df, filters, facility_uids)


def render_patient_filter_controls(patient_df, container=None, context="default"):
    """Simple filter controls for patient data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
    key_suffix = f"_{context}"

    print(f"\n🔧 DEBUG render_patient_filter_controls for {context}")

    # Time Period options
    time_options = [
        "All Time",  # Default/first option
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

    print(f"   Selected quick_range: {filters['quick_range']}")

    # Get REAL VALID dates from patient dataframe
    min_date, max_date = _get_patient_date_range(patient_df)
    print(f"   REAL VALID data date range: {min_date} to {max_date}")

    # Ensure min_date is not earlier than 2000 (for date_input constraints)
    if min_date < datetime.date(2000, 1, 1):
        min_date = datetime.date(2000, 1, 1)

    # Ensure max_date is not later than 2035 (for date_input constraints)
    if max_date > datetime.date(2035, 12, 31):
        max_date = datetime.date(2035, 12, 31)

    # Set dates based on selection
    if filters["quick_range"] == "All Time":
        print("   All Time selected")
        filters["start_date"] = min_date
        filters["end_date"] = max_date

    elif filters["quick_range"] == "Custom Range":
        print("   Custom Range selected")

        # For Custom Range, use CURRENT YEAR as default (Jan to Dec of current year)
        today = datetime.date.today()
        default_start = datetime.date(today.year, 1, 1)  # Jan 1 of current year
        default_end = datetime.date(today.year, 12, 31)  # Dec 31 of current year

        # Ensure defaults are within min/max bounds
        if default_start < min_date:
            default_start = min_date
        if default_end > max_date:
            default_end = max_date

        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=default_start,  # Current year Jan 1
                min_value=datetime.date(2000, 1, 1),
                max_value=max_date,
                key=f"start_date{key_suffix}",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=default_end,  # Current year Dec 31
                min_value=min_date,
                max_value=datetime.date(2035, 12, 31),
                key=f"end_date{key_suffix}",
            )
        print(f"   Custom range: {filters['start_date']} to {filters['end_date']}")

    else:
        print(f"   Predefined range: {filters['quick_range']}")
        temp_df = patient_df.copy()

        # Use the REAL data for date calculation
        _df_for_dates = temp_df.copy()
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date
        print(f"   Calculated range: {filters['start_date']} to {filters['end_date']}")

    # Aggregation Level
    print(f"\n   Getting available aggregations...")
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )
    print(f"   Available aggregations: {available_aggregations}")

    default_index = (
        available_aggregations.index("Monthly")
        if "Monthly" in available_aggregations
        else 0
    )

    # Get current value from session state if exists
    current_period_label = st.session_state.get("period_label", "Monthly")
    print(f"   Current period_label in session_state: {current_period_label}")

    if current_period_label in available_aggregations:
        default_index = available_aggregations.index(current_period_label)
        print(f"   Found in available aggregations, index: {default_index}")

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",
    )

    print(f"   Selected period_label: {filters['period_label']}")

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
        "selected_kpi", "Institutional Maternal Death Rate (per 100,000 births)"
    )

    # Add show_table option
    filters["show_table"] = container.checkbox(
        "📊 Show Data Table",
        value=False,
        key=f"show_table{key_suffix}",
        help="Display the data table below the chart",
    )

    print(f"   Filters created: {list(filters.keys())}")
    print(f"🔧 DEBUG render_patient_filter_controls END\n")

    return filters


def _get_patient_date_range(patient_df):
    """Get REAL min/max dates from patient dataframe - FILTERS OUT INVALID DATES"""
    import datetime
    import pandas as pd

    if patient_df.empty:
        today = datetime.date.today()
        return today, today

    print(f"🔧 DEBUG _get_patient_date_range: Finding REAL date range")

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
                    # Filter out dates that are clearly invalid (like 1970-01-01)
                    for date_val in valid_dates:
                        try:
                            date_dt = date_val.to_pydatetime()
                            year = date_dt.year
                            # Only accept dates from year 2000 onward (reasonable for your data)
                            if year >= 2000 and year <= 2030:  # Reasonable range
                                all_valid_dates.append(date_val)
                        except:
                            continue

                    if all_valid_dates:
                        print(f"   Found {len(valid_dates)} dates in column '{col}'")
            except Exception as e:
                print(f"   Error parsing '{col}': {str(e)}")

    if all_valid_dates:
        # Find the REAL minimum and maximum VALID dates
        min_date = min(all_valid_dates)
        max_date = max(all_valid_dates)

        # Convert to date objects
        if hasattr(min_date, "date"):
            min_date = min_date.date()
        if hasattr(max_date, "date"):
            max_date = max_date.date()

        print(f"   REAL VALID date range: {min_date} to {max_date}")
        return min_date, max_date

    # If no valid dates found, use current year
    today = datetime.date.today()
    current_year_start = datetime.date(today.year, 1, 1)
    current_year_end = datetime.date(today.year, 12, 31)

    print(
        f"⚠️ No valid dates found, using current year: {current_year_start} to {current_year_end}"
    )
    return current_year_start, current_year_end


def apply_patient_filters(patient_df, filters, facility_uids=None):
    """Apply filters to patient dataframe - COMPLETE UPDATED VERSION with proper UID filtering"""
    if patient_df.empty:
        print("⚠️ apply_patient_filters: Patient dataframe is empty")
        return patient_df

    df = patient_df.copy()

    # DEBUG: Check initial count
    print(f"\n{'='*60}")
    print(f"🔧 DEBUG apply_patient_filters START:")
    print(f"{'='*60}")
    print(f"   Original rows: {len(df)}")
    print(f"   Quick range: {filters.get('quick_range', 'Not specified')}")

    # ✅ IMPORTANT: For "All Time", don't apply any date filtering at all!
    quick_range = filters.get("quick_range", "")
    skip_date_filtering = quick_range == "All Time"

    if skip_date_filtering:
        print("   ⚠️ 'All Time' selected - SKIPPING date filtering")
    else:
        print("   Date filtering WILL be applied")

    # ✅ CRITICAL: Verify orgUnit column exists
    if "orgUnit" not in df.columns:
        print("❌ ERROR: 'orgUnit' column not found in data!")
        print(f"   Available columns: {list(df.columns)}")
        return df

    # ✅ Apply facility filter FIRST using UIDs
    print(f"\n🏥 FACILITY FILTERING (USING UIDs):")
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        facility_count = len(df)
        print(f"   Filtering by {len(facility_uids)} UIDs")
        print(f"   UIDs to filter: {facility_uids}")

        # Check which UIDs exist in the data
        existing_uids = [uid for uid in facility_uids if uid in df["orgUnit"].values]
        missing_uids = [uid for uid in facility_uids if uid not in df["orgUnit"].values]

        print(f"   UIDs found in data: {len(existing_uids)}/{len(facility_uids)}")
        if missing_uids:
            print(f"   UIDs NOT in data: {missing_uids}")

        # Filter by UIDs
        facility_mask = df["orgUnit"].isin(facility_uids)
        df = df[facility_mask].copy()
        print(
            f"   After UID filtering: {len(df)} rows (removed {facility_count - len(df)})"
        )

        # Show facility breakdown
        if not df.empty:
            facility_breakdown = df["orgUnit"].value_counts()
            print(f"   Facility breakdown:")
            for uid, count in facility_breakdown.items():
                # Get facility name if available
                facility_name = "Unknown"
                if "orgUnit_name" in df.columns:
                    names = df[df["orgUnit"] == uid]["orgUnit_name"].unique()
                    if len(names) > 0:
                        facility_name = names[0]
                print(f"     {uid} ({facility_name}): {count} rows")
    else:
        print(f"   No facility filter applied")

    # ✅ CRITICAL FIX: Check for 'combined_date' first (from regional.py)
    if "combined_date" in df.columns:
        print("   ✅ Using 'combined_date' column from regional.py")
        df["event_date"] = pd.to_datetime(df["combined_date"], errors="coerce")
    elif "event_date" not in df.columns:
        # Normalize dates if no event_date column
        print("   Normalizing dates...")
        df = normalize_patient_dates(df)
        print(f"   After normalization: {'event_date' in df.columns}")

    # ✅ CRITICAL FIX: Ensure event_date exists and is datetime
    if "event_date" in df.columns:
        valid_dates = df["event_date"].notna().sum()
        print(f"   Valid event dates: {valid_dates}/{len(df)}")
        if valid_dates > 0:
            min_date = df["event_date"].min()
            max_date = df["event_date"].max()
            print(f"   Date range in data: {min_date} to {max_date}")
    else:
        print(f"❌ ERROR: 'event_date' column missing!")
        print(f"   Available columns: {list(df.columns)}")
        return df

    # ✅ CRITICAL FIX: Only apply date filtering if NOT "All Time"
    if (
        not skip_date_filtering
        and filters.get("start_date")
        and filters.get("end_date")
    ):
        print(f"\n📅 APPLYING DATE FILTERING:")
        print(f"   Filter start_date: {filters.get('start_date')}")
        print(f"   Filter end_date: {filters.get('end_date')}")

        try:
            # Convert filter dates to datetime
            start_dt = pd.Timestamp(filters["start_date"])
            end_dt = pd.Timestamp(filters["end_date"])

            print(
                f"   Applying date filter range: {start_dt.date()} to {end_dt.date()}"
            )

            # Ensure event_date exists
            if "event_date" in df.columns:
                valid_dates = df["event_date"].notna().sum()
                if valid_dates > 0:
                    print(
                        f"   Original data date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}"
                    )

            # Filter by date range
            before_filter = len(df)
            date_mask = (df["event_date"] >= start_dt) & (df["event_date"] <= end_dt)
            df = df[date_mask].copy()

            after_filter = len(df)
            rows_removed = before_filter - after_filter
            print(
                f"   After date filtering: {after_filter} rows (removed {rows_removed})"
            )

        except Exception as e:
            print(f"❌ Error applying date filter: {str(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
    else:
        print(f"\n📅 SKIPPING DATE FILTERING (All Time or no dates specified)")
        print(f"   Keeping all {len(df)} rows without date filtering")

    # ✅ CRITICAL FIX 4: Ensure period_label is in filters
    print(f"\n📊 PERIOD ASSIGNMENT:")
    if "period_label" not in filters:
        print(f"❌ apply_patient_filters: period_label not in filters!")
        print(f"   Available keys in filters: {list(filters.keys())}")
        # Try to get from session state
        if "period_label" in st.session_state:
            filters["period_label"] = st.session_state.period_label
            print(f"   Got period_label from session_state: {filters['period_label']}")
        else:
            filters["period_label"] = "Monthly"
            print(f"⚠️ Using default period_label: Monthly")
    else:
        print(f"   Using period_label from filters: '{filters['period_label']}'")

    print(
        f"🔧 DEBUG apply_patient_filters: Using period_label = '{filters['period_label']}'"
    )

    # ✅ CRITICAL FIX 5: Save to session state for prepare_data_for_trend_chart
    st.session_state.period_label = filters["period_label"]
    print(
        f"🔧 DEBUG apply_patient_filters: Saved to session_state: '{filters['period_label']}'"
    )

    # Also ensure filters dict is in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters.update(filters)

    # ✅ CRITICAL FIX 6: Assign period AFTER date filtering
    print(f"\n🔧 DEBUG apply_patient_filters: Calling assign_period with:")
    print(f"   df shape before assign_period: {df.shape}")
    print(f"   date_col: 'event_date'")
    print(f"   period_label: '{filters['period_label']}'")
    print(f"   Rows with valid dates: {df['event_date'].notna().sum()}/{len(df)}")

    try:
        # Only assign period to rows with valid dates
        valid_date_rows = df["event_date"].notna().sum()
        if valid_date_rows > 0:
            print(f"   Assigning periods to {valid_date_rows} rows with valid dates")
            df = assign_period(df, "event_date", filters["period_label"])

            # ✅ DEBUG: Check what periods were created
            if "period_display" in df.columns:
                unique_periods = df["period_display"].unique()
                print(f"✅ assign_period created {len(unique_periods)} unique periods:")

                # Show period distribution
                period_counts = df["period_display"].value_counts().sort_index()
                for period, count in period_counts.items():
                    print(f"   {period}: {count} rows")

                # If only one period, that's OK for monthly filtering
                if len(unique_periods) == 1:
                    print(
                        f"   Note: Single period {unique_periods[0]} - this is expected for monthly filters"
                    )
            else:
                print(f"❌ assign_period did not create 'period_display' column!")
                print(f"   Columns after assign_period: {list(df.columns)}")
        else:
            print(f"⚠️ No valid dates to assign periods to")
            # Create empty period columns
            df["period_display"] = ""
            df["period_sort"] = 0
            print(f"   Created empty period columns")

    except Exception as e:
        print(f"❌ ERROR in assign_period: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        # Create empty period columns as fallback
        df["period_display"] = ""
        df["period_sort"] = 0

    print(f"\n✅ apply_patient_filters complete.")
    print(f"   Final shape: {df.shape}")
    print(
        f"   Unique TEI IDs: {df['tei_id'].nunique() if 'tei_id' in df.columns else 'N/A'}"
    )
    print(
        f"   Unique orgUnits: {df['orgUnit'].nunique() if 'orgUnit' in df.columns else 'N/A'}"
    )

    # Show sample of filtered data
    if not df.empty:
        print(f"   Sample of filtered data:")
        if "event_date" in df.columns and df["event_date"].notna().any():
            print(f"     Dates: {df['event_date'].head(3).tolist()}")
        if "period_display" in df.columns:
            print(f"     Periods: {df['period_display'].head(3).tolist()}")
        if "orgUnit" in df.columns:
            uids = df["orgUnit"].unique()[:3]
            print(f"     OrgUnit UIDs: {list(uids)}")
        if "orgUnit_name" in df.columns:
            facilities = df["orgUnit_name"].unique()[:3]
            print(f"     Facility names: {list(facilities)}")

    print(f"{'='*60}")
    print(f"🔧 DEBUG apply_patient_filters END")
    print(f"{'='*60}\n")

    return df


def get_period_data_for_kpi(kpi_selection, patient_df, facility_uids):
    """Get period-based data for a specific KPI from patient data"""
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

    # ⚠️ CRITICAL: DON'T filter out rows without dates here!
    # The updated prepare_data_for_trend_chart will handle fallback to enrollment_date
    # This ensures we keep all TEI IDs, even those without event dates

    # Sort by period
    if "period_sort" in df_with_periods.columns:
        df_with_periods = df_with_periods.sort_values("period_sort")

    return df_with_periods


def compute_kpi_for_period(kpi_selection, period_df, facility_uids):
    """Compute KPI for a specific period using patient data"""
    if period_df.empty:
        return _get_default_kpi_data(kpi_selection)

    # Use the get_numerator_denominator_for_kpi function (which now includes enrollment date fallback)
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
    # Map KPIs to their relevant date columns
    if kpi_selection in [
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
        "Early Postnatal Care (PNC) Coverage (%)",
    ]:
        # PNC-related KPIs use PNC date
        for col in [PNC_DATE_COL, "event_date_postpartum_care"]:
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
        for col in [DELIVERY_DATE_COL, "event_date_delivery_summary"]:
            if col in df.columns:
                return col

    elif kpi_selection == "ARV Prophylaxis Rate (%)":
        # ARV-related KPIs might use various dates
        for col in [
            PNC_DATE_COL,
            DELIVERY_DATE_COL,
            "event_date_postpartum_care",
            "event_date_delivery_summary",
        ]:
            if col in df.columns:
                return col

    elif kpi_selection == "Low Birth Weight (LBW) Rate (%)":
        # LBW uses birth weight from delivery
        for col in [DELIVERY_DATE_COL, "event_date_delivery_summary"]:
            if col in df.columns:
                return col

    # Fallback to any event_date column
    for col in df.columns:
        if "event_date" in col.lower():
            return col

    # Last resort - check for enrollment_date
    if ENROLLMENT_DATE_COL in df.columns:
        return ENROLLMENT_DATE_COL

    return "event_date"

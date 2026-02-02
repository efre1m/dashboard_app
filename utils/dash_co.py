import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime
import logging

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
from utils.kpi_missing_cod import (
    compute_missing_cod_kpi,
    render_missing_cod_trend_chart,
    render_missing_cod_facility_comparison_chart,
    render_missing_cod_region_comparison_chart,
    get_numerator_denominator_for_missing_cod,
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
from utils.kpi_missing_bo import (
    compute_missing_bo_kpi,
    render_missing_bo_trend_chart,
    render_missing_bo_facility_comparison_chart,
    render_missing_bo_region_comparison_chart,
    get_numerator_denominator_for_missing_bo,
)

from utils.kpi_assisted import (
    compute_assisted_kpi,
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

from utils.kpi_missing_md import (
    compute_missing_md_kpi,
    render_missing_md_trend_chart,
    render_missing_md_facility_comparison_chart,
    render_missing_md_region_comparison_chart,
    get_numerator_denominator_for_missing_md,
)
from utils.kpi_admitted_mothers import (
    compute_admitted_mothers_kpi,
    render_admitted_mothers_trend_chart,
    render_admitted_mothers_facility_comparison_chart,
    render_admitted_mothers_region_comparison_chart,
    get_numerator_denominator_for_admitted_mothers,
)
from utils.kpi_episiotomy import (
    compute_episiotomy_kpi,
    render_episiotomy_trend_chart,
    render_episiotomy_facility_comparison_chart,
    render_episiotomy_region_comparison_chart,
    get_numerator_denominator_for_episiotomy,
)
from utils.kpi_antipartum_compl import (
    compute_antipartum_compl_kpi,
    render_antipartum_compl_trend_chart,
    render_antipartum_compl_facility_comparison_chart,
    render_antipartum_compl_region_comparison_chart,
    render_complication_type_pie_chart,
    get_numerator_denominator_for_antipartum_compl,
)
from utils.kpi_postpartum_compl import (
    compute_postpartum_compl_kpi,
    render_postpartum_compl_trend_chart,
    render_postpartum_compl_facility_comparison_chart,
    render_postpartum_compl_region_comparison_chart,
    render_postpartum_complication_type_pie_chart,
    get_numerator_denominator_for_postpartum_compl,
)
from utils.kpi_missing_postpartum import (
    compute_missing_postpartum_kpi,
    render_missing_postpartum_trend_chart,
    render_missing_postpartum_facility_comparison_chart,
    render_missing_postpartum_region_comparison_chart,
    get_numerator_denominator_for_missing_postpartum,
)
from utils.kpi_missing_antepartum import (
    compute_missing_antepartum_kpi,
    render_missing_antepartum_trend_chart,
    render_missing_antepartum_facility_comparison_chart,
    render_missing_antepartum_region_comparison_chart,
    get_numerator_denominator_for_missing_antepartum,
)
from utils.kpi_missing_uterotonic import (
    compute_missing_uterotonic_kpi,
    render_missing_uterotonic_trend_chart,
    render_missing_uterotonic_facility_comparison_chart,
    render_missing_uterotonic_region_comparison_chart,
    get_numerator_denominator_for_missing_uterotonic,
)

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
        "denominator_name": "Total Newborns",
    },
    "Early Postnatal Care (PNC) Coverage (%)": {
        "title": "Early PNC Coverage (%)",
        "numerator_name": "Early PNC (≤48 hrs)",
        "denominator_name": "Total Deliveries",
    },
    "Maternal Death Rate (per 100,000)": {
        "title": "Maternal Death Rate (per 100,000)",
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
        "denominator_name": "Total Deliveries",
    },
    "ARV Prophylaxis Rate (%)": {
        "title": "ARV Prophylaxis Rate (%)",
        "numerator_name": "ARV Cases",
        "denominator_name": "HIV-Exposed Infants",
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
        "title": "Missing Mode of Delivery Documentation Rate (%)",
        "numerator_name": "Deliveries with Missing MD",
        "denominator_name": "Total Deliveries",
    },
    "Missing Birth Outcome": {
        "title": "Missing Birth Outcome Documentation Rate (%)",
        "numerator_name": "Missing Birth Outcomes",
        "denominator_name": "Total Newborns",
    },
    "Missing Condition of Discharge": {
        "title": "Missing Condition of Discharge Documentation Rate (%)",
        "numerator_name": "Missing Condition of Discharge",
        "denominator_name": "Total Mothers",
    },
    "Admitted Mothers": {
        "title": "Total Admitted Mothers",
        "value_name": "Admitted Mothers",
    },
    "Episiotomy Rate (%)": {
        "title": "Episiotomy Rate (%)",
        "numerator_name": "Episiotomy Cases",
        "denominator_name": "Total Vaginal Deliveries",
    },
    "Antepartum Complications Rate (%)": {
        "title": "Antepartum Complications Rate (%)",
        "numerator_name": "Complication Cases",
        "denominator_name": "Total Deliveries",
    },
    "Postpartum Complications Rate (%)": {
        "title": "Postpartum Complications Rate (%)",
        "numerator_name": "Complication Cases",
        "denominator_name": "Total Deliveries",
    },
    "Missing Obstetric Condition at Delivery": {
        "title": "Missing Obstetric Condition at Delivery Documentation Rate (%)",
        "numerator_name": "Missing Obstetric Condition at Delivery",
        "denominator_name": "Total Deliveries",
    },
    "Missing Obstetric Complications Diagnosis": {
        "title": "Missing Obstetric Complications Diagnosis Documentation Rate (%)",
        "numerator_name": "Missing Obstetric Complications Diagnosis",
        "denominator_name": "Total Deliveries",
    },
    "Missing Uterotonics Given at Delivery": {
        "title": "Missing Uterotonics Given at Delivery Documentation Rate (%)",
        "numerator_name": "Missing Uterotonics Given at Delivery",
        "denominator_name": "Total Deliveries",
    },
}

# KPI Column Requirements - What each KPI actually needs
KPI_COLUMN_REQUIREMENTS = {
    "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "fp_counseling_and_method_provided_pp_postpartum_care",
    ],
    "Stillbirth Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_outcome_delivery_summary",
        "birth_outcome_newborn_delivery_summary",
        "birth_outcome_newborn_2_delivery_summary",
        "birth_outcome_newborn_3_delivery_summary",
        "birth_outcome_newborn_4_delivery_summary",
        "number_of_newborns_delivery_summary",
        "other_number_of_newborns_delivery_summary",
    ],
    "Early Postnatal Care (PNC) Coverage (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "date_stay_pp_postpartum_care",
    ],
    "Maternal Death Rate (per 100,000)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "condition_of_discharge_discharge_summary",
    ],
    "C-Section Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "mode_of_delivery_maternal_delivery_summary",
    ],
    "Postpartum Hemorrhage (PPH) Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "obstetric_condition_at_delivery_delivery_summary",
    ],
    "Delivered women who received uterotonic (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "uterotonics_given_delivery_summary",
    ],
    "ARV Prophylaxis Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "hiv_result_delivery_summary",
        "arv_rx_for_newborn_by_type_pp_postpartum_care",
        "birth_outcome_delivery_summary",
        "number_of_newborns_delivery_summary",
        "other_number_of_newborns_delivery_summary",
    ],
    "Assisted Delivery Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "instrumental_delivery_form",
        "mode_of_delivery_maternal_delivery_summary",
    ],
    "Normal Vaginal Delivery (SVD) Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "mode_of_delivery_maternal_delivery_summary",
    ],
    "Missing Mode of Delivery": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "mode_of_delivery_maternal_delivery_summary",
        "instrumental_delivery_form",
    ],
    "Missing Birth Outcome": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_outcome_delivery_summary",
        "birth_outcome_newborn_delivery_summary",
        "birth_outcome_newborn_2_delivery_summary",
        "birth_outcome_newborn_3_delivery_summary",
        "birth_outcome_newborn_4_delivery_summary",
        "number_of_newborns_delivery_summary",
        "other_number_of_newborns_delivery_summary",
    ],
    "Missing Condition of Discharge": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "enrollment_status",
        "condition_of_discharge_discharge_summary",
    ],
    "Admitted Mothers": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
    ],
    "Episiotomy Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "episiotomy_performed_delivery_summary",
        "mode_of_delivery_maternal_delivery_summary",
        "instrumental_delivery_form",
    ],
    "Antepartum Complications Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "obstetric_complications_diagnosis",
    ],
    "Postpartum Complications Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "obstetric_condition_at_delivery_delivery_summary",
    ],
    "Missing Obstetric Condition at Delivery": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "obstetric_condition_at_delivery_delivery_summary",
    ],
    "Missing Obstetric Complications Diagnosis": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "obstetric_complications_diagnosis",
    ],
    "Missing Uterotonics Given at Delivery": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "uterotonics_given_delivery_summary",
    ],
}


def get_kpi_filtered_dataframe(df, kpi_name):
    """
    Filter DataFrame to only include columns needed for the selected KPI
    """
    if df is None or df.empty:
        return df

    if kpi_name not in KPI_COLUMN_REQUIREMENTS:
        # If KPI not in mapping, return all columns
        return df.copy()

    # Get required columns for this KPI
    required_columns = KPI_COLUMN_REQUIREMENTS[kpi_name]

    # Find which columns actually exist in the DataFrame
    available_columns = []
    for col in required_columns:
        if col in df.columns:
            available_columns.append(col)

    # Always include these essential columns if they exist
    essential_cols = ["orgUnit", "tei_id", "enrollment_date"]
    for col in essential_cols:
        if col in df.columns and col not in available_columns:
            available_columns.append(col)

    # If we have no columns after filtering, return original DataFrame
    if not available_columns:
        return df.copy()

    # Create filtered DataFrame
    filtered_df = df[available_columns].copy()

    # Log the optimization
    original_size = len(df.columns)
    filtered_size = len(filtered_df.columns)
    print(f"✅ KPI Filter: {kpi_name}")
    print(f"   Before: {original_size} columns")
    print(f"   After: {filtered_size} columns")
    print(
        f"   Reduction: {((original_size - filtered_size) / original_size * 100):.1f}%"
    )

    return filtered_df


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
    "Maternal Death Rate (per 100,000)",
    "C-Section Rate (%)",
    "Postpartum Hemorrhage (PPH) Rate (%)",
    "Delivered women who received uterotonic (%)",
    "ARV Prophylaxis Rate (%)",
    "Assisted Delivery Rate (%)",
    "Normal Vaginal Delivery (SVD) Rate (%)",
    "Missing Mode of Delivery",
    "Missing Birth Outcome",
    "Missing Condition of Discharge",
    "Admitted Mothers",
    "Episiotomy Rate (%)",
    "Antepartum Complications Rate (%)",
    "Postpartum Complications Rate (%)",
    "Missing Obstetric Condition at Delivery",
    "Missing Obstetric Complications Diagnosis",
    "Missing Uterotonics Given at Delivery",
]


def render_kpi_tab_navigation():
    """Render professional tab navigation for KPI selection"""

    st.markdown(
        """
    <style>
    /* Ultra-Compact Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1px 4px !important;
        font-size: 10px !important;
        height: 22px !important;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1a6790 !important;
        font-weight: 800 !important;
        padding: 0.05rem 0.3rem !important;
        font-size: 10px !important;
        margin-bottom: 0px !important;
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
        font-weight: 600 !important;
        padding: 0.05rem 0.3rem !important;
        font-size: 10px !important;
        margin-bottom: 0px !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e9ecef !important;
        color: #495057 !important;
        border-color: #ced4da !important;
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
    # KPI Grouping for Tab Navigation - UPDATED NAMES
    # Tabs: Enrollment, Care, Data Quality, Complication, Mortality
    
    # Initialize session state for KPI selection
    if "selected_kpi" not in st.session_state:
        st.session_state.selected_kpi = "Admitted Mothers"

    # Create main KPI group tabs
    tab_enroll, tab_care, tab_dq, tab_compl, tab_mort = st.tabs(
        [
            "Enrollment",
            "Care",
            "Data Quality",
            "Complication",
            "Mortality",
        ]
    )

    selected_kpi = st.session_state.selected_kpi

    # 1. Enrollment Tab
    with tab_enroll:
        # Match "Admitted Newborns" style - use columns(5) to make button smaller
        cols = st.columns(5)
        with cols[0]:
            if st.button(
                "Admitted Mothers",
                key="admitted_mothers_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Admitted Mothers" else "secondary"),
            ):
                selected_kpi = "Admitted Mothers"

    # 2. Care Tab
    # Order: SVD, Assisted, Episiotomy, C-Section, Postpartum Comp, Uterotonic, FP, PNC, ARV
    with tab_care:
        # First Row
        cols1 = st.columns(4)
        with cols1[0]:
            if st.button(
                "SVD",
                key="svd_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Normal Vaginal Delivery (SVD) Rate (%)" else "secondary"),
            ):
                selected_kpi = "Normal Vaginal Delivery (SVD) Rate (%)"
        
        with cols1[1]:
            if st.button(
                "Assisted Delivery",
                key="assisted_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Assisted Delivery Rate (%)" else "secondary"),
            ):
                selected_kpi = "Assisted Delivery Rate (%)"

        with cols1[2]:
            if st.button(
                "Episiotomy Rate",
                key="episiotomy_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Episiotomy Rate (%)" else "secondary"),
            ):
                selected_kpi = "Episiotomy Rate (%)"

        with cols1[3]:
            if st.button(
                "C-Section Rate",
                key="csection_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "C-Section Rate (%)" else "secondary"),
            ):
                selected_kpi = "C-Section Rate (%)"

        # Second Row
        cols2 = st.columns(3)
        with cols2[0]:
            if st.button(
                "Delivered women who received uterotonic",
                key="uterotonic_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Delivered women who received uterotonic (%)" else "secondary"),
            ):
                selected_kpi = "Delivered women who received uterotonic (%)"
        
        with cols2[1]:
            if st.button(
                "FP Counseling and Acceptance",
                key="contraceptive_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)" else "secondary"),
            ):
                selected_kpi = "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"

        with cols2[2]:
            if st.button(
                "Early Postnatal Care (PNC) Coverage",
                key="pnc_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Early Postnatal Care (PNC) Coverage (%)" else "secondary"),
            ):
                selected_kpi = "Early Postnatal Care (PNC) Coverage (%)"
        
        # Third Row
        cols3 = st.columns(1)
        with cols3[0]:
            if st.button(
                "ARV Prophylaxis Rate",
                key="arv_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "ARV Prophylaxis Rate (%)" else "secondary"),
            ):
                selected_kpi = "ARV Prophylaxis Rate (%)"

    # 3. Data Quality Tab
    with tab_dq:
        cols = st.columns(3)
        with cols[0]:
            if st.button(
                "Missing Mode of Delivery",
                key="missing_md_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Mode of Delivery" else "secondary"),
            ):
                selected_kpi = "Missing Mode of Delivery"
        
        with cols[1]:
            if st.button(
                "Missing Birth Outcome",
                key="missing_bo_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Birth Outcome" else "secondary"),
            ):
                selected_kpi = "Missing Birth Outcome"
        
        with cols[2]:
            if st.button(
                "Missing Condition of Discharge",
                key="missing_cod_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Condition of Discharge" else "secondary"),
            ):
                selected_kpi = "Missing Condition of Discharge"

        # Second Row for Data Quality
        cols_dq2 = st.columns(3)
        with cols_dq2[0]:
            if st.button(
                "Missing Obstetric Condition at Delivery",
                key="missing_postpartum_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Obstetric Condition at Delivery" else "secondary"),
            ):
                selected_kpi = "Missing Obstetric Condition at Delivery"
        
        with cols_dq2[1]:
            if st.button(
                "Missing Obstetric Complications Diagnosis",
                key="missing_antepartum_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Obstetric Complications Diagnosis" else "secondary"),
            ):
                selected_kpi = "Missing Obstetric Complications Diagnosis"

        with cols_dq2[2]:
            if st.button(
                "Missing Uterotonics Given at Delivery",
                key="missing_uterotonic_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Missing Uterotonics Given at Delivery" else "secondary"),
            ):
                selected_kpi = "Missing Uterotonics Given at Delivery"

    # 4. Complication Tab
    # Antepartum Complication -> Postpartum Complication -> PPH
    with tab_compl:
        cols = st.columns(3)
        with cols[0]:
            if st.button(
                "Antepartum Complication",
                key="antipartum_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Antepartum Complications Rate (%)" else "secondary"),
            ):
                selected_kpi = "Antepartum Complications Rate (%)"
        
        with cols[1]:
            if st.button(
                "Postpartum Complication",
                key="postpartum_compl_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Postpartum Complications Rate (%)" else "secondary"),
            ):
                selected_kpi = "Postpartum Complications Rate (%)"

        with cols[2]:
            if st.button(
                "Postpartum Hemorrhage (PPH) Rate",
                key="pph_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Postpartum Hemorrhage (PPH) Rate (%)" else "secondary"),
            ):
                selected_kpi = "Postpartum Hemorrhage (PPH) Rate (%)"

    # 5. Mortality Tab
    with tab_mort:
        cols = st.columns(2)
        with cols[0]:
            if st.button(
                "Mortality Rate per 100,000",
                key="maternal_death_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Maternal Death Rate (per 100,000)" else "secondary"),
            ):
                selected_kpi = "Maternal Death Rate (per 100,000)"
        
        with cols[1]:
            if st.button(
                "Stillbirth Rate",
                key="stillbirth_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Stillbirth Rate (%)" else "secondary"),
            ):
                selected_kpi = "Stillbirth Rate (%)"

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
    """Render the trend chart with KPI-specific program stage dates - FIXED VERSION"""

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in patient_df.columns:
        for col in patient_df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                patient_df = patient_df.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in patient_df.columns:
        st.error("❌ 'orgUnit' column not found in data. Cannot filter by UIDs.")
        return

    # NO SPECIAL HANDLING NEEDED! Missing MD will use the same flow

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_kpi_filtered_dataframe(patient_df, kpi_selection)

    # Get KPI configuration for labels
    kpi_config = get_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)

    if kpi_df.empty:
        st.info("⚠️ No data available for trend analysis.")
        return

    # Apply UID filter
    working_df = kpi_df.copy()
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_kpi(kpi_selection)

    # Get date range filters from session state
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # CRITICAL FIX: Use prepare_data_for_trend_chart to get data filtered by KPI-specific dates AND date range
    prepared_df, date_column_used = prepare_data_for_trend_chart(
        working_df, kpi_selection, facility_uids, date_range_filters
    )

    if prepared_df.empty:
        st.warning(
            f"⚠️ No data available for {kpi_selection} using its specific program stage date: '{date_column_used}'"
        )
        return

    # Ensure period columns exist with proper formatting
    if "period_display" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            prepared_df["period_display"] = (
                prepared_df["event_date"].dt.strftime("%b-%y").str.capitalize()
            )  # Format as "Sep-25"
        else:
            prepared_df["period_display"] = "Period"

    if "period_sort" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            try:
                prepared_df["period_sort"] = pd.to_datetime(
                    prepared_df["event_date"]
                ).dt.strftime("%Y%m")
            except:
                prepared_df["period_sort"] = prepared_df.index
        else:
            prepared_df["period_sort"] = prepared_df.index

    # Get unique periods in order
    unique_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Create period data - SIMPLIFIED APPROACH
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period from prepared_df (already filtered by KPI-specific dates)
        period_df = prepared_df[prepared_df["period_display"] == period_display]

        if not period_df.empty:
            # Get TEI IDs from this period (already filtered by KPI-specific dates)
            period_tei_ids = period_df["tei_id"].dropna().unique()

            if not period_tei_ids.size:
                continue

            # CRITICAL: Get the original data for these TEI IDs, but FILTERED by KPI-specific date
            # Use working_df (which already has facility filter applied)
            period_patient_data = working_df.copy()

            # Filter by TEI IDs AND KPI-specific date
            if date_column_used and date_column_used in period_patient_data.columns:
                period_patient_data[date_column_used] = pd.to_datetime(
                    period_patient_data[date_column_used], errors="coerce"
                )

                # Get TEI IDs that have the KPI-specific date
                tei_ids_with_date = period_patient_data[
                    period_patient_data[date_column_used].notna()
                ]["tei_id"].unique()

                # Intersection: TEI IDs that are in our period AND have KPI-specific date
                valid_tei_ids = set(period_tei_ids) & set(tei_ids_with_date)

                if valid_tei_ids:
                    period_patient_data = period_patient_data[
                        period_patient_data["tei_id"].isin(valid_tei_ids)
                        & period_patient_data[date_column_used].notna()
                    ].copy()
                else:
                    # No valid TEI IDs with KPI-specific date
                    period_data.append(
                        {
                            "period": period_display,
                            "period_display": period_display,
                            "period_sort": period_sort,
                            "value": 0,
                            "numerator": 0,
                            "denominator": 0,
                        }
                    )
                    continue
            else:
                # Fallback: just filter by TEI IDs
                period_patient_data = period_patient_data[
                    period_patient_data["tei_id"].isin(period_tei_ids)
                ].copy()

            # Compute KPI using date-filtered data - THIS WILL NOW WORK FOR ALL KPIs INCLUDING MISSING MD!
            numerator, denominator, _ = get_numerator_denominator_for_kpi(
                period_patient_data, kpi_selection, facility_uids, date_range_filters
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

    # Render the chart WITH TABLE - ADD MISSING MD HERE!
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
        elif kpi_selection == "Episiotomy Rate (%)":
            render_episiotomy_trend_chart(
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
        elif kpi_selection == "Antepartum Complications Rate (%)":
            render_antipartum_compl_trend_chart(
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
        elif kpi_selection == "Postpartum Complications Rate (%)":
            from utils.kpi_postpartum_compl import render_postpartum_compl_trend_chart
            render_postpartum_compl_trend_chart(
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
        # ADD MISSING MD HERE - IT'S JUST ANOTHER KPI!
        elif kpi_selection == "Missing Mode of Delivery":
            render_missing_md_trend_chart(
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
        elif kpi_selection == "Missing Birth Outcome":
            render_missing_bo_trend_chart(
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

        elif kpi_selection == "Missing Condition of Discharge":
            render_missing_cod_trend_chart(
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
        # In render_trend_chart_section function, replace the Admitted Mothers section:
        elif kpi_selection == "Admitted Mothers":
            # For Admitted Mothers, we need to handle it differently since it's not a percentage
            # Get date range filters
            date_range_filters = {}
            if "filters" in st.session_state:
                date_range_filters = {
                    "start_date": st.session_state.filters.get("start_date"),
                    "end_date": st.session_state.filters.get("end_date"),
                }

            # Create period data for Admitted Mothers
            period_data = []

            for _, row in unique_periods.iterrows():
                period_display = row["period_display"]
                period_sort = row["period_sort"]

                # Get data for this period from prepared_df
                period_df = prepared_df[prepared_df["period_display"] == period_display]

                if not period_df.empty:
                    # Get TEI IDs from this period
                    period_tei_ids = period_df["tei_id"].dropna().unique()

                    if not period_tei_ids.size:
                        continue

                    # Get data for these TEI IDs
                    period_patient_data = working_df.copy()

                    # Filter by TEI IDs AND enrollment date
                    if (
                        date_column_used
                        and date_column_used in period_patient_data.columns
                    ):
                        period_patient_data[date_column_used] = pd.to_datetime(
                            period_patient_data[date_column_used], errors="coerce"
                        )

                        tei_ids_with_date = period_patient_data[
                            period_patient_data[date_column_used].notna()
                        ]["tei_id"].unique()

                        valid_tei_ids = set(period_tei_ids) & set(tei_ids_with_date)

                        if valid_tei_ids:
                            period_patient_data = period_patient_data[
                                period_patient_data["tei_id"].isin(valid_tei_ids)
                                & period_patient_data[date_column_used].notna()
                            ].copy()
                        else:
                            period_data.append(
                                {
                                    "period": period_display,
                                    "period_display": period_display,
                                    "period_sort": period_sort,
                                    "value": 0,
                                    "numerator": 0,
                                    "denominator": 0,
                                }
                            )
                            continue
                    else:
                        period_patient_data = period_patient_data[
                            period_patient_data["tei_id"].isin(period_tei_ids)
                        ].copy()

                    # For Admitted Mothers, use the new function
                    from utils.kpi_admitted_mothers import (
                        get_numerator_denominator_for_admitted_mothers,
                    )

                    numerator, denominator, _ = (
                        get_numerator_denominator_for_admitted_mothers(
                            period_patient_data, facility_uids, date_range_filters
                        )
                    )

                    # For Admitted Mothers, the value is the count (numerator)
                    # NOT a percentage calculation
                    value = float(numerator)  # Just use the count as value

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
                st.info("⚠️ No period data available for Admitted Mothers chart.")
                return

            # Create DataFrame
            group = pd.DataFrame(period_data)
            group = group.sort_values("period_sort")

            # Render the chart
            render_admitted_mothers_trend_chart(
                group,
                "period_display",
                "value",
                "Total Admitted Mothers Trend",
                bg_color,
                text_color,
                display_names,
                "Admitted Mothers",
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
    show_chart=True,  # New argument
):
    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }
    """Render comparison charts for both national and regional views WITH TABLES using KPI-specific dates"""

    df_to_use = filtered_patients if filtered_patients is not None else patient_df

    if df_to_use is None or df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_kpi_filtered_dataframe(df_to_use, kpi_selection)
    df_to_use = kpi_df  # Use the filtered DataFrame

    # REMOVE THIS - Missing MD should have comparison charts too!
    # if kpi_selection == "Missing Mode of Delivery":
    #     st.info("⚠️ Comparison chart not available for Missing Mode of Delivery.")
    #     return

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

    # CRITICAL FIX: Prepare data using prepare_data_for_trend_chart for EACH facility/region
    # This ensures KPI-specific dates are used throughout

    if comparison_mode == "facility":
        comparison_data = []

        # For each facility, prepare its own trend data
        for facility_uid, facility_name in zip(facility_uids, display_names):
            # Filter data for this specific facility
            facility_df = df_to_use[df_to_use["orgUnit"] == facility_uid].copy()

            if facility_df.empty:
                # Add zero entries for all periods
                comparison_data.append(
                    {
                        "period_display": "All Periods",
                        "orgUnit": facility_uid,
                        "orgUnit_name": facility_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Prepare facility-specific data with KPI-specific dates AND date range filters
            prepared_df, _ = prepare_data_for_trend_chart(
                facility_df, kpi_selection, [facility_uid], date_range_filters
            )

            if prepared_df.empty:
                # Add zero entry
                comparison_data.append(
                    {
                        "period_display": "All Periods",
                        "orgUnit": facility_uid,
                        "orgUnit_name": facility_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Group by period for this facility
            for period_display, period_group in prepared_df.groupby("period_display"):
                if not period_group.empty:
                    # Get TEI IDs from this period (already filtered by KPI-specific dates)
                    period_tei_ids = period_group["tei_id"].dropna().unique()

                    # Get data for these TEI IDs from the ORIGINAL facility data
                    # but ONLY if they have the KPI-specific date
                    if date_column and date_column in facility_df.columns:
                        period_data = facility_df.copy()
                        period_data[date_column] = pd.to_datetime(
                            period_data[date_column], errors="coerce"
                        )
                        # Filter by TEI IDs AND KPI-specific date
                        period_data = period_data[
                            period_data["tei_id"].isin(period_tei_ids)
                            & period_data[date_column].notna()
                        ].copy()
                    else:
                        period_data = facility_df[
                            facility_df["tei_id"].isin(period_tei_ids)
                        ].copy()

                    # Compute KPI for this period - SPECIAL HANDLING FOR ADMITTED MOTHERS
                    if kpi_selection == "Admitted Mothers":
                        # Use Admitted Mothers specific function
                        from utils.kpi_admitted_mothers import (
                            get_numerator_denominator_for_admitted_mothers,
                        )

                        numerator, denominator, _ = (
                            get_numerator_denominator_for_admitted_mothers(
                                period_data, [facility_uid], date_range_filters
                            )
                        )

                        # For Admitted Mothers, value is the count (numerator)
                        value = float(numerator)
                    else:
                        # For all other KPIs, use the standard function
                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            period_data,
                            kpi_selection,
                            [facility_uid],
                            date_range_filters,
                        )

                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    comparison_data.append(
                        {
                            "period_display": period_display,
                            "orgUnit": facility_uid,
                            "orgUnit_name": facility_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not comparison_data:
            st.info("⚠️ No comparison data available.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # Ensure proper column names
        if "orgUnit_name" in comparison_df.columns:
            comparison_df = comparison_df.rename(columns={"orgUnit_name": "Facility"})

        # RENDER TABLE IF CHART IS HIDDEN
        if not show_chart:
            st.markdown(f"### {chart_title} - Comparison Table")
            
            # Format value separation depending on KPI
            cols_to_show = ["period_display", "Facility", "value", "numerator", "denominator"]
            
            # Simplified dataframe for display
            display_df = comparison_df[cols_to_show].copy()
            
            # Rename for display
            display_df = display_df.rename(columns={
                "period_display": "Period",
                "value": "Value (%)" if "Rate" in kpi_selection or "%" in kpi_selection else "Value",
                "numerator": numerator_label,
                "denominator": denominator_label
            })
            
            st.dataframe(display_df, use_container_width=True)
            return

        # Call the appropriate chart function - ADD MISSING MD HERE!
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
            )
        else:
            # For general KPIs using generic chart function (e.g. Missing Data)
            # Check if specialized function exists, else use generic
            if kpi_selection == "Missing Mode of Delivery":
                 render_missing_md_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )
            elif kpi_selection == "Missing Birth Outcome":
                 render_missing_bo_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )
            elif kpi_selection == "Missing Condition of Discharge":
                 # Use generic for now or implement specific
                 pass
            elif kpi_selection == "Admitted Mothers":
                 render_admitted_mothers_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )
            elif kpi_selection == "Episiotomy Rate (%)":
                render_episiotomy_facility_comparison_chart(
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
                )
            elif kpi_selection == "Antepartum Complications Rate (%)":
                render_antipartum_compl_facility_comparison_chart(
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
                )
            elif kpi_selection == "Postpartum Complications Rate (%)":
                render_postpartum_compl_facility_comparison_chart(
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
                )
            elif kpi_selection == "Missing Obstetric Condition at Delivery":
                render_missing_postpartum_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )
            elif kpi_selection == "Missing Obstetric Complications Diagnosis":
                render_missing_antepartum_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )
            elif kpi_selection == "Missing Uterotonics Given at Delivery":
                render_missing_uterotonic_facility_comparison_chart(
                    df=comparison_df,
                    period_col="period_display",
                    value_col="value",
                    title=chart_title,
                    bg_color=bg_color,
                    text_color=text_color,
                    facility_names=display_names,
                    facility_uids=facility_uids,
                )

    elif comparison_mode == "region" and facilities_by_region:
        # REGIONAL LOGIC - placeholder for now
        pass


    elif comparison_mode == "region" and is_national:
        # Region comparison - similar fix needed
        region_data = []

        # Get facility UIDs for each region
        region_facility_mapping = {}
        for region_name in region_names:
            region_facility_mapping[region_name] = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

        for region_name in region_names:
            region_facility_uids = region_facility_mapping.get(region_name, [])
            if not region_facility_uids:
                continue

            # Get data for this region
            region_df = df_to_use[
                df_to_use["orgUnit"].isin(region_facility_uids)
            ].copy()

            if region_df.empty:
                continue

            # Prepare region-specific data with KPI-specific dates AND date range filters
            prepared_df, _ = prepare_data_for_trend_chart(
                region_df, kpi_selection, region_facility_uids, date_range_filters
            )

            if prepared_df.empty:
                continue

            # Group by period for this region
            for period_display, period_group in prepared_df.groupby("period_display"):
                if not period_group.empty:
                    # Get TEI IDs from this period (already filtered by KPI-specific dates)
                    period_tei_ids = period_group["tei_id"].dropna().unique()

                    # Get data for these TEI IDs
                    if date_column and date_column in region_df.columns:
                        period_data = region_df.copy()
                        period_data[date_column] = pd.to_datetime(
                            period_data[date_column], errors="coerce"
                        )
                        period_data = period_data[
                            period_data["tei_id"].isin(period_tei_ids)
                            & period_data[date_column].notna()
                        ].copy()
                    else:
                        period_data = region_df[
                            region_df["tei_id"].isin(period_tei_ids)
                        ].copy()

                    # Compute KPI for this period - SPECIAL HANDLING FOR ADMITTED MOTHERS
                    if kpi_selection == "Admitted Mothers":
                        # Use Admitted Mothers specific function
                        from utils.kpi_admitted_mothers import (
                            get_numerator_denominator_for_admitted_mothers,
                        )

                        numerator, denominator, _ = (
                            get_numerator_denominator_for_admitted_mothers(
                                period_data, region_facility_uids, date_range_filters
                            )
                        )

                        # For Admitted Mothers, value is the count (numerator)
                        value = float(numerator)
                    else:
                        # For all other KPIs, use the standard function
                        numerator, denominator, _ = get_numerator_denominator_for_kpi(
                            period_data,
                            kpi_selection,
                            region_facility_uids,
                            date_range_filters,
                        )

                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    region_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not region_data:
            st.info("⚠️ No comparison data available for regions.")
            return

        region_df = pd.DataFrame(region_data)

        # Call the appropriate region comparison function - ADD MISSING MD HERE!
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
        elif kpi_selection == "Episiotomy Rate (%)":
            render_episiotomy_region_comparison_chart(
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
        elif kpi_selection == "Antepartum Complications Rate (%)":
            render_antipartum_compl_region_comparison_chart(
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
        elif kpi_selection == "Postpartum Complications Rate (%)":
            render_postpartum_compl_region_comparison_chart(
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

        elif kpi_selection == "Missing Mode of Delivery":
            render_missing_md_region_comparison_chart(
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
        elif kpi_selection == "Missing Birth Outcome":
            render_missing_bo_region_comparison_chart(
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
        elif kpi_selection == "Missing Condition of Discharge":
            render_missing_cod_region_comparison_chart(
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
        elif kpi_selection == "Admitted Mothers":
            render_admitted_mothers_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name="Admitted Mothers",
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

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_kpi_filtered_dataframe(patient_df, kpi_selection)

    if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
        # REMOVED PIE CHART FOR PPH AS REQUESTED
        pass
    elif kpi_selection == "Delivered women who received uterotonic (%)":
        render_uterotonic_type_pie_chart(kpi_df, facility_uids, bg_color, text_color)
    elif kpi_selection == "Antepartum Complications Rate (%)":
        render_complication_type_pie_chart(kpi_df, facility_uids, bg_color, text_color)
    elif kpi_selection == "Postpartum Complications Rate (%)":
        render_postpartum_complication_type_pie_chart(kpi_df, facility_uids, bg_color, text_color)
    elif kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
        # Render FP Distribution Pie Chart
        st.session_state.kpi_cache = st.session_state.get("kpi_cache", {})
        # We need to manually compute distribution if not already available in cache or re-compute it here
        # Actually, let's just use the data frame
        from utils.kpi_utils import compute_fp_distribution
        dist = compute_fp_distribution(patient_df, facility_uids)
        
        if dist:
            # Create a simple pie chart logic here for FP
            pie_data = [{"Method": k, "Count": v} for k, v in dist.items()]
            pie_df = pd.DataFrame(pie_data)
            
            if not pie_df.empty:
                import plotly.express as px
                st.markdown('<div style="text-align: center; font-weight: bold;">Distribution of FP Methods</div>', unsafe_allow_html=True)
                fig = px.pie(
                    pie_df, values="Count", names="Method", height=400,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, key="fp_viz")


def normalize_patient_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single datetime column 'enrollment_date' exists for patient data"""
    if df.empty:
        if "enrollment_date" not in df.columns:
            df["enrollment_date"] = pd.Series(dtype="datetime64[ns]")
        return df

    df = df.copy()

    # Get current KPI to use the right date column
    current_kpi = st.session_state.get(
        "selected_kpi", "Institutional Maternal Death Rate (%)"
    )
    from utils.kpi_utils import get_relevant_date_column_for_kpi

    kpi_date_column = get_relevant_date_column_for_kpi(current_kpi)

    # Try KPI-specific date column first
    if "enrollment_date" in df.columns:
        df["enrollment_date"] = pd.to_datetime(df["enrollment_date"], errors="coerce")
        logging.info("✅ normalize_patient_dates: Using enrollment_date")
    elif kpi_date_column and kpi_date_column in df.columns:
        df["enrollment_date"] = pd.to_datetime(df[kpi_date_column], errors="coerce")
        logging.info(
            f"✅ normalize_patient_dates: Using KPI-specific '{kpi_date_column}'"
        )
    elif "combined_date" in df.columns:
        df["enrollment_date"] = pd.to_datetime(df["combined_date"], errors="coerce")
        logging.info("✅ normalize_patient_dates: Using combined_date")
    else:
        # Look for program stage event dates
        program_stage_date_columns = [
            col
            for col in df.columns
            if col.startswith("event_date_") or col == "event_date"
        ]

        for col in program_stage_date_columns:
            try:
                df["enrollment_date"] = pd.to_datetime(df[col], errors="coerce")
                if not df["enrollment_date"].isna().all():
                    logging.info(
                        f"✅ normalize_patient_dates: Using {col} for enrollment_date"
                    )
                    break
            except:
                continue

    # If still no date found, use current date
    if "enrollment_date" not in df.columns or df["enrollment_date"].isna().all():
        df["enrollment_date"] = pd.Timestamp.now().normalize()
        logging.warning(
            "⚠️ normalize_patient_dates: No valid date column found, using current date"
        )

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
        # Use the FULL patient dataframe directly
        start_date, end_date = get_date_range(patient_df, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date
        logging.info(
            f"📅 Filter dates set: {start_date} to {end_date} for '{filters['quick_range']}'"
        )

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
    """Get REAL min/max dates from patient dataframe - HARDCODED ALL TIME"""
    import datetime
    import logging

    # ALWAYS return the full 2000-2035 range
    all_time_start = datetime.date(2000, 1, 1)
    all_time_end = datetime.date(2035, 12, 31)

    logging.info(
        f"📅 _get_patient_date_range: Using ALL TIME range: {all_time_start} to {all_time_end}"
    )

    return all_time_start, all_time_end


def apply_patient_filters(patient_df, filters, facility_uids=None):
    """Apply filters to patient dataframe - USING KPI-SPECIFIC DATES - FIXED FOR ALL FACILITIES"""
    if patient_df.empty:
        return patient_df

    df = patient_df.copy()

    # Get current KPI selection from session state
    current_kpi = st.session_state.get(
        "selected_kpi", "Institutional Maternal Death Rate (%)"
    )

    # Get the SPECIFIC date column for this KPI
    from utils.kpi_utils import get_relevant_date_column_for_kpi

    kpi_date_column = get_relevant_date_column_for_kpi(current_kpi)

    logging.info(f"🔍 apply_patient_filters for KPI: {current_kpi}")
    logging.info(f"   - Using date column: {kpi_date_column}")
    logging.info(f"   - Quick range: {filters.get('quick_range', 'N/A')}")
    logging.info(f"   - Facility UIDs: {facility_uids if facility_uids else 'All'}")

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

    # CRITICAL FIX: Apply facility filter ONLY if we have specific facility UIDs
    # Skip if "All Facilities" or empty
    if facility_uids and "orgUnit" in df.columns:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        # Check if this is "All Facilities" or actual facility UIDs
        # "All Facilities" should have actual UID strings, not the text "All Facilities"
        is_all_facilities = (
            len(facility_uids) == 0
            or facility_uids == ["All Facilities"]
            or (len(facility_uids) == 1 and facility_uids[0] == "All Facilities")
        )

        if not is_all_facilities:
            facility_mask = df["orgUnit"].isin(facility_uids)
            df = df[facility_mask].copy()
            logging.info(f"   - After facility filter: {len(df)} patients")
        else:
            logging.info("   - Skipping facility filter (All Facilities selected)")
    else:
        logging.info("   - No facility filter applied")

    # CRITICAL FIX: Always use normalize_patient_dates for consistency
    # This ensures we have a proper enrollment_date column
    df = normalize_patient_dates(df)

    # Use enrollment_date for all filtering (KPI-specific filtering happens in get_numerator_denominator_for_kpi)
    logging.info(f"   ✅ Using normalized enrollment_date for date filtering")

    # STEP 1: Check if we should filter by date
    should_filter_by_date = (
        not skip_date_filtering  # Not "All Time"
        and filters.get("start_date")  # Has start date
        and filters.get("end_date")  # Has end date
        and "enrollment_date" in df.columns  # Has enrollment_date column
    )

    if should_filter_by_date:
        try:
            # STEP 2: Get the start and end dates from filters
            start_date = pd.Timestamp(filters["start_date"])
            end_date = pd.Timestamp(filters["end_date"])

            logging.info(
                f"🔍 DATE FILTER: Filtering from {start_date.date()} to {end_date.date()}"
            )

            # STEP 3: Use enrollment_date for filtering (already created by normalize_patient_dates)
            date_column_to_use = "enrollment_date"

            # Check if enrollment_date has data
            enrollment_dates_valid = df["enrollment_date"].notna().sum()
            logging.info(
                f"📅 Using 'enrollment_date' column ({enrollment_dates_valid} valid dates)"
            )

            # STEP 4: Apply the date filter
            if date_column_to_use and enrollment_dates_valid > 0:
                # Make sure the date column is properly formatted as datetime
                df[date_column_to_use] = pd.to_datetime(
                    df[date_column_to_use], errors="coerce"
                )

                # Create a mask: keep rows where date is between start_date and end_date
                date_is_after_start = df[date_column_to_use] >= start_date
                date_is_before_end = df[date_column_to_use] <= end_date
                date_mask = date_is_after_start & date_is_before_end

                # Apply the filter
                df = df[date_mask].copy()

                # Log how many patients are left
                patients_before = len(patient_df)
                patients_after = len(df)
                patients_lost = patients_before - patients_after
                logging.info(
                    f"✅ DATE FILTER COMPLETE: {patients_after} patients remain (lost {patients_lost})"
                )
            else:
                logging.warning(
                    "⚠️ No valid dates found for filtering. Keeping all patients."
                )

        except Exception as error:
            logging.error(f"❌ ERROR in date filtering: {error}")
            # If there's an error, keep all patients (don't filter)
    else:
        logging.info("⏰ Skipping date filtering (All Time selected or no dates)")

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

    # Assign period AFTER date filtering - using enrollment_date
    try:
        valid_date_rows = df["enrollment_date"].notna().sum()
        if valid_date_rows > 0:
            df = assign_period(df, "enrollment_date", filters["period_label"])
            logging.info(
                f"   📊 Assigned {filters['period_label']} periods: {valid_date_rows} valid dates"
            )
        else:
            # Create empty period columns
            df["period_display"] = ""
            df["period_sort"] = 0
            logging.warning("   ⚠️ No valid dates for period assignment")
    except Exception as e:
        # Create empty period columns as fallback
        df["period_display"] = ""
        df["period_sort"] = 0
        logging.error(f"   ❌ Error assigning periods: {e}")

    # OPTIMIZATION: Filter for current KPI
    # Filter DataFrame to only include columns needed for this KPI
    if current_kpi in KPI_COLUMN_REQUIREMENTS:
        df = get_kpi_filtered_dataframe(df, current_kpi)
        logging.info(f"   🔍 KPI filter applied for '{current_kpi}'")

    logging.info(f"🔄 apply_patient_filters: Final result - {len(df)} patients")
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
    """Compute KPI for a specific period using patient data - ULTRA SIMPLE VERSION"""
    if period_df.empty:
        return _get_default_kpi_data(kpi_selection)

    # For ALL KPIs, use get_numerator_denominator_for_kpi
    numerator, denominator, value = get_numerator_denominator_for_kpi(
        period_df, kpi_selection, facility_uids
    )

    # Return SIMPLE, consistent structure
    return {
        "value": float(value),
        "numerator": int(numerator),
        "denominator": int(denominator),
        "kpi_name": kpi_selection,
    }


def _get_default_kpi_data(kpi_selection):
    """Get default empty data"""
    return {"value": 0.0, "numerator": 0, "denominator": 0}


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
